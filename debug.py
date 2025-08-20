from models.modules import ConvolutionModule
from models.attention import MultiHeadSelfAttentionModule
from models.encoder import ConformerBlock
import torch
from torch import nn
import json

vocab_size = 100
n_layers = 2
d_model = 32
d_ff = 64
h = 4
p_dropout = 0.1
in_features = 80
batch_size = 2
seq_len_enc = 15
seq_len_dec = 27

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==== Tạo dữ liệu đầu vào ====
src = torch.randn(batch_size, seq_len_enc, in_features)                 # encoder input
src_d_model = torch.randn(batch_size, seq_len_enc, d_model) 
tgt = torch.randint(0, vocab_size, (batch_size, seq_len_dec))            # decoder input

# ==== Tạo mask ====
src_mask = torch.ones(batch_size, seq_len_enc)           # [B, 1, M, T]
tgt_mask = torch.ones(batch_size, 1, seq_len_dec, seq_len_dec)           # [B, 1, M, M]


def test_convolution_module():
    dim_model = in_features
    dim_expand = 64
    kernel_size = 3
    Pdrop = p_dropout
    stride = 1
    padding = 'causal'

    conv_module = ConvolutionModule(dim_model, dim_expand, kernel_size, Pdrop, stride, padding)
    
    # Forward pass
    output = conv_module(src)
    
    print("Convolution Module Output Shape:", output.shape)  # Expected shape: [batch_size, seq_len_enc, dim_expand]

def test_attention_module():
    mha = MultiHeadSelfAttentionModule(h,d_model, p_dropout, max_pos_encoding=5000)
    
    output, _, _ = mha(src_d_model, src_mask)
    print("Attention Module Output Shape:", output.shape)  # Expected shape: [batch_size, seq_len_enc, d_model]

def test_conformer_block():
    from models.encoder import ConformerBlock
    
    dim_model = 32
    dim_expand = 64
    ff_ratio = 2
    num_heads = h
    kernel_size = 3
    Pdrop = p_dropout
    conv_stride = 1
    att_stride = 1
    padding = 'causal'

    conformer_block = ConformerBlock(dim_model, dim_expand, ff_ratio, num_heads, kernel_size, Pdrop, conv_stride, att_stride, padding)
    
    output, attention, _ = conformer_block(src_d_model, src_mask)
    
    print("Conformer Block Output Shape:", output.shape)  # Expected shape: [batch_size, seq_len_enc, dim_model]
    print("Attention Weights Shape:", attention.shape)     # Expected shape: [batch_size, num_heads, seq_len_enc, seq_len_enc]

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def test_encoder():
    from models.encoder import ConformerEncoder
    config = load_json('/home/anhkhoa/Conformer/configs/config_local.json')
    
    
    encoder = ConformerEncoder(config['encoder_params']).to(device)
    
    output, x_len = encoder(src.to(device), x_len=torch.tensor([seq_len_enc] * batch_size).to(device))
    
    print("Encoder Output Shape:", output.shape)  # Expected shape: [batch_size, seq_len_enc, d_model]
    print("Encoder Output Lengths:", x_len)       # Expected lengths after subsampling


def test_model():
    from models.model import ConformerTransducer
    config = load_json('/home/anhkhoa/Conformer/configs/config_local.json')
    
    model = ConformerTransducer(config).to(device)
    
    inputs = src.to(device)
    inputs_length = torch.tensor([seq_len_enc] * batch_size).to(device)
    targets = tgt.to(device)
    targets_length = torch.tensor([seq_len_dec] * batch_size).to(device)
    
    logits = model(inputs, inputs_length, targets, targets_length)
    
    print("Model Logits Shape:", logits.shape)  # Expected shape: [batch_size, seq_len_dec, vocab_size]

if __name__ == "__main__":
    # test_convolution_module()
    # test_attention_module()
    # test_conformer_block()
    # test_encoder()
    test_model()



