import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import librosa
from speechbrain.lobes.features import Fbank
import speechbrain as sb


# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r", encoding= 'utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def get_blank_token(self):
        return self.stoi["<blank>"]
    def __len__(self):
        return len(self.vocab)


class AudioPreprocessing(nn.Module):

    """Audio Preprocessing

    Computes mel-scale log filter banks spectrogram

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT frame size, creates n_fft // 2 + 1 frequency bins.
        win_length_ms: FFT window length in ms, must be <= n_fft
        hop_length_ms: length of hop between FFT windows in ms
        n_mels: number of mel filter banks
        normalize: whether to normalize mel spectrograms outputs
        mean: training mean
        std: training std

    Shape:
        Input: (batch_size, audio_len)
        Output: (batch_size, n_mels, audio_len // hop_length + 1)
    
    """

    def __init__(self, sample_rate, n_fft, win_length_ms, hop_length_ms, n_mels, normalize, mean, std):
        super(AudioPreprocessing, self).__init__()
        self.win_length = int(sample_rate * win_length_ms) // 1000
        self.hop_length = int(sample_rate * hop_length_ms) // 1000
        self.Spectrogram = torchaudio.transforms.Spectrogram(n_fft, self.win_length, self.hop_length)
        self.MelScale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min=0, f_max=8000, n_stft=n_fft // 2 + 1)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x, x_len):

        # Short Time Fourier Transform (B, T) -> (B, n_fft // 2 + 1, T // hop_length + 1)
        x = self.Spectrogram(x)

        # Mel Scale (B, n_fft // 2 + 1, T // hop_length + 1) -> (B, n_mels, T // hop_length + 1)
        x = self.MelScale(x)
        
        # Energy log, autocast disabled to prevent float16 overflow
        x = (x.float() + 1e-9).log().type(x.dtype)

        # Compute Sequence lengths 
        if x_len is not None:
            x_len = torch.div(x_len, self.hop_length, rounding_mode='floor') + 1

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        return x, x_len

def stack_context(x, left=3, right=1):
    """x: (T, D) -> (T, (left+1+right)*D) | pad biên bằng replicate."""
    T, D = x.shape
    pads = []
    for off in range(-left, right + 1):
        idx = np.clip(np.arange(T) + off, 0, T - 1)
        pads.append(x[idx])
    return np.concatenate(pads, axis=1)

def subsample(x, base_hop_ms=10, target_hop_ms=30):
    stride = target_hop_ms // base_hop_ms
    return x[::stride]


class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, apply_spec_augment=True):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()
        self.apply_spec_augment = apply_spec_augment
        self.fbank = Fbank(
            sample_rate=16000,
            n_mels=80,
            n_fft=512,
            win_length=25,
        )


    def __len__(self):
        return len(self.data)

    def get_fbank(self, waveform, sample_rate=16000):
        y, sr = librosa.load(waveform, sr=sample_rate)
        win_length = int(0.025 * sr)   # 25 ms
        hop_length = int(0.010 * sr)   # 10 ms
        # n_fft = next power of 2 >= win_length
        n_fft = 1
        while n_fft < win_length:
            n_fft *= 2

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=40, n_fft=n_fft,
            win_length=win_length, hop_length=hop_length,
            window='hann', power=2.0, center=True
        )
        # log-mel (dB)
        x = librosa.power_to_db(S, ref=np.max).T   # (T, 40)
        
        mu = x.mean(axis=0, keepdims=True)
        sg = x.std(axis=0, keepdims=True) + 1e-8
        x = (x - mu) / sg
        x = stack_context(x, left=3, right=1) 
        return torch.tensor(subsample(x, 10, 30))

    # def extract_from_path(self, wave_path):
    #     sig  = sb.dataio.dataio.read_audio(wave_path)
    #     return self.get_fbank(sig.unsqueeze(0))

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"] + [self.pad_token], dtype=torch.long)
        tokens = torch.tensor(current_item["encoded_text"], dtype=torch.long)
        fbank = self.get_fbank(wav_path).float()  # [T, 512]

        return {
            "text": encoded_text,
            "fbank": fbank,
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,
            "tokens": tokens,
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def speech_collate_fn(batch):
    decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    text_mask=calculate_mask(text_lens, padded_texts.size(1) + 1)

    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask
    }

