import torch
import torch.nn as nn
from .modules import ConvolutionModule, FeedForwardBlock, ConvolutionResidual, AttentionResidual, Conv2dSubsampling, get_mask_from_lens,PositionalEncoding
from .attention import MultiHeadSelfAttentionModule
import torchaudio

class SpecAugment(nn.Module):

    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, spec_augment, mF, F, mT, pS):
        super(SpecAugment, self).__init__()
        self.spec_augment = spec_augment
        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, x, x_len):

        # Spec Augment
        if self.spec_augment:
        
            # Frequency Masking
            for _ in range(self.mF):
                x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)

            # Time Masking
            for b in range(x.size(0)):
                T = int(self.pS * x_len[b])
                for _ in range(self.mT):
                    x[b:b+1, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(x[b:b+1, :, :x_len[b]])

        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim_model, dim_expand, ff_ratio, num_heads, kernel_size, Pdrop, conv_stride, att_stride, padding):
        super(ConformerBlock, self).__init__()
        self.conv_module = ConvolutionModule(dim_model, dim_expand, kernel_size, Pdrop, conv_stride, "causal" if padding == 'causal' else 'same')
        self.ffn_1 = FeedForwardBlock(dim_model, dim_model * ff_ratio,  Pdrop, act = 'swish')
        self.ffn_2 = FeedForwardBlock(dim_expand, dim_expand * ff_ratio, Pdrop, act = 'swish')
        self.multihead_attention = MultiHeadSelfAttentionModule(num_heads, dim_model, Pdrop, max_pos_encoding = 5000)
        # self.conv_residual = ConvolutionResidual(dim_model, dim_expand, kernel_size, conv_stride)
        self.norm = nn.LayerNorm(dim_expand, eps=1e-6)
        # self.atten_residual = AttentionResidual(att_stride)
        self.stride = conv_stride * att_stride

    def forward(self, x, mask=None, hidden=None):
        x = x + 1/2 * self.ffn_1(x)
        x_att, attention, _ = self.multihead_attention(x, mask, hidden)
        x = x  + x_att
        x  = x + self.conv_module(x)
        x = x + 1/2 * self.ffn_2(x)
        x = self.norm(x)
        return x, attention, hidden


class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super(ConformerEncoder, self).__init__()
        self.d_model = config['d_model']
        self.dim_expand = config['dim_expand']
        self.ff_ratio = config['ff_ratio']
        self.num_heads = config['num_heads']
        self.kernel_size = config['kernel_size']
        self.p_dropout = config['p_dropout']
        self.conv_stride = config['conv_stride']
        self.att_stride = config['att_stride']
        self.padding = config['padding']
        self.n_layers = config['n_layers']

        self.conv_subsampling = Conv2dSubsampling(
            num_layers=config['conv_subsampling']['num_layers'],
            filters=config['conv_subsampling']['filters'],
            kernel_size=config['conv_subsampling']['kernel_size'],
            norm = config['conv_subsampling']['norm'],
            act = config['conv_subsampling']['act'],
        )

        self.layers = nn.ModuleList([
            ConformerBlock(
                dim_model=self.d_model,
                dim_expand=self.dim_expand,
                ff_ratio=self.ff_ratio,
                num_heads=self.num_heads,
                kernel_size=self.kernel_size,
                Pdrop=self.p_dropout,
                conv_stride=self.conv_stride,
                att_stride=self.att_stride,
                padding=self.padding
            ) for _ in range(self.n_layers)
        ])
        self.linear = nn.Linear(config['in_features'], self.d_model)
        self.dropout = nn.Dropout(self.p_dropout)
        # self.pe = PositionalEncoding(d_model=self.d_model)
        self.projected = nn.Linear(self.d_model, config['output_size'])
        self.spec_augment = SpecAugment(
            spec_augment=config['spec_augment']['spec_augment'],
            mF=config['spec_augment']['mF'],
            F=config['spec_augment']['F'],
            mT=config['spec_augment']['mT'],
            pS=config['spec_augment']['pS']
        )
    def forward(self, x, x_len = None, training = True):
        if training:
            x = self.spec_augment(x, x_len)
        x, x_len = self.conv_subsampling(x, x_len)

        # print(x_len.shape)
        # print(x.size(2))
        mask = get_mask_from_lens(x_len, x.size(1)).to(x.device)
        
        
        x = self.linear(x)
        x = self.dropout(x)
        # x = self.pe(x)

        for layer in self.layers:
            x, attention, _ = layer(x, mask)
        
        x = self.projected(x)
        return x, x_len 

def build_encoder(config):
    return ConformerEncoder(config['encoder_params'])