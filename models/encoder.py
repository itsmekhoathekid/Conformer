import torch
import torch.nn as nn
from .modules import ConvolutionModule, FeedForwardBlock, ConvolutionResidual, AttentionResidual, Conv2dSubsampling, get_mask_from_lens,PositionalEncoding
from .attention import MultiHeadSelfAttentionModule

class ConformerBlock(nn.Module):
    def __init__(self, dim_model, dim_expand, ff_ratio, num_heads, kernel_size, Pdrop, conv_stride, att_stride, padding):
        super(ConformerBlock, self).__init__()
        self.conv_module = ConvolutionModule(dim_model, dim_expand, kernel_size, Pdrop, conv_stride, "causal" if padding == 'causal' else 'same')
        self.ffn_1 = FeedForwardBlock(dim_model, dim_model * ff_ratio,  Pdrop, act = 'swish')
        self.ffn_2 = FeedForwardBlock(dim_expand, dim_expand * ff_ratio, Pdrop, act = 'swish')
        self.multihead_attention = MultiHeadSelfAttentionModule(num_heads, dim_model, Pdrop, max_pos_encoding = 5000)
        self.conv_residual = ConvolutionResidual(dim_model, dim_expand, kernel_size, conv_stride)
        self.norm = nn.LayerNorm(dim_expand, eps=1e-6)
        self.atten_residual = AttentionResidual(att_stride)
        self.stride = conv_stride * att_stride

    def forward(self, x, mask=None, hidden=None):
        x = x + 1/2 * self.ffn_1(x)
        x_att, attention, _ = self.multihead_attention(x, mask, hidden)
        x = self.atten_residual(x_att) + x_att
        x  = self.conv_residual(x) + self.conv_module(x)
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
        self.pe = PositionalEncoding(d_model=self.d_model)
        self.projected = nn.Linear(self.d_model, config['output_size'])
    
    def forward(self, x, x_len = None):
        x, x_len = self.conv_subsampling(x, x_len)

        # print(x_len.shape)
        # print(x.size(2))
        mask = get_mask_from_lens(x_len, x.size(2)).to(x.device)
        
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.pe(x)

        for layer in self.layers:
            x, attention, _ = layer(x, mask)
        
        x = self.projected(x)
        return x, x_len 

def build_encoder(config):
    return ConformerEncoder(config['encoder_params'])