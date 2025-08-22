import torch
import torch.nn as nn
from .modules import ConvolutionModule, FeedForwardBlock, ConvolutionResidual, AttentionResidual, Conv2dSubsampling, get_mask_from_lens,PositionalEncoding, ResidualConnection, FeedForwardModule
from .attention import MultiHeadSelfAttentionModule
import torchaudio
from typing import Optional, Callable, Type, List

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
    def __init__(self, dim_model, dim_expand, ff_ratio, num_heads, kernel_size, Pdrop, conv_stride, att_stride, padding, attention_type):
        super(ConformerBlock, self).__init__()
        self.conv_module = ConvolutionModule(dim_model, dim_expand, kernel_size, Pdrop, conv_stride, "causal" if padding == 'causal' else 'same')
        self.ffn_1 = FeedForwardModule(dim_model, dim_model * ff_ratio,  Pdrop)
        self.ffn_2 = FeedForwardModule(dim_model, dim_model * ff_ratio,  Pdrop)
        self.multihead_attention = MultiHeadSelfAttentionModule(num_heads, dim_model, Pdrop, max_pos_encoding = 5000, attention_type=attention_type)
        self.conv_residual = ConvolutionResidual(dim_model, dim_expand, kernel_size, conv_stride)
        # self.norm1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_expand, eps=1e-6)
        # self.atten_residual = AttentionResidual(att_stride)
        self.stride = conv_stride * att_stride
        self.dropout = nn.Dropout(Pdrop)
        self.residual_modules = nn.ModuleList([
            ResidualConnection(dim_model, Pdrop, 0.5),
            ResidualConnection(dim_expand, Pdrop, 1),
            ResidualConnection(dim_model, Pdrop, 1),
            ResidualConnection(dim_expand, Pdrop, 0.5)
        ])
    def forward(self, x, mask=None, hidden=None):
        x = self.residual_modules[0](x, lambda x: self.ffn_1(x))
        x = self.residual_modules[1](x, lambda x: self.multihead_attention(x, mask, hidden))
        x = self.residual_modules[2](x, lambda x: self.conv_residual(x))
        x = self.residual_modules[3](x, lambda x: self.ffn_2(x))
        x = self.norm2(x)
        return x

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        residual: bool = False,
        conv_module: Type[nn.Module] = nn.Conv2d,
        activation: Callable = nn.LeakyReLU,  # 👉 Dùng LeakyReLU
        norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv_stride = stride if i == num_layers - 1 else 1
            conv = conv_module(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                dilation=dilation,
                padding=(kernel_size // 2)
            )
            layers.append(conv)
            if norm:
                layers.append(norm(out_channels))  # Gọi instance
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)
        self.residual = residual

        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm(out_channels) if norm else nn.Identity(),
                nn.Dropout(dropout),
            )
        elif residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x, x_len):
        B, C, T, F = x.shape
        residual_input = x  

        for layer in self.main:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - x_len
                data_len = x_len
                x_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                T = out_T

        if self.residual:
            shortcut = self.shortcut(residual_input)  # 👉 fix chỗ này
            x = x + shortcut

        return x, x_len


class ConvolutionFrontEnd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        num_layers_per_block: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        residuals: List[bool],
        activation: Callable = nn.LeakyReLU, 
        norm: Optional[Callable] = nn.BatchNorm2d, 
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []

        for i in range(num_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels[i],
                num_layers=num_layers_per_block,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                residual=residuals[i],
                activation=activation,
                norm=norm,
                dropout=dropout
            )
            blocks.append(block)
            in_channels = out_channels[i]

        self.model = nn.ModuleList(blocks)

    def forward(self, x, x_len):
        x = x.unsqueeze(1)
        for i, block in enumerate(self.model):
            x, x_len = block(x, x_len)
        x = x.transpose(1,2)
        # print(x.shape)
        x = x.reshape(x.size(0), x.size(1), -1)  # Flatten the last two dimensions
        # print(x.shape)
        return x, x_len

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

        if config['conv_type'] == 1:
            self.conv_subsampling = Conv2dSubsampling(
                num_layers=config['conv_subsampling']['num_layers'],
                filters=config['conv_subsampling']['filters'],
                kernel_size=config['conv_subsampling']['kernel_size'],
                norm = config['conv_subsampling']['norm'],
                act = config['conv_subsampling']['act'],
            )
        else:
            self.conv_subsampling = ConvolutionFrontEnd(
                in_channels=1,
                num_blocks=3,
                num_layers_per_block=2,
                out_channels=[8, 16, 32],
                kernel_sizes=[3, 3, 3],
                strides=[1, 2, 2],
                residuals=[True, True, True],
                activation=nn.ReLU,        
                norm=nn.BatchNorm2d,            
                dropout=0.1,
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
                padding=self.padding,
                attention_type = config["attention_type"]
            ) for _ in range(self.n_layers)
        ])
        self.linear = nn.Linear(config['in_features'], self.d_model)
        self.dropout = nn.Dropout(self.p_dropout)
        
        self.pe = PositionalEncoding(d_model=self.d_model) if config["attention_type"] == "mha" else None

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
        x = self.pe(x) if self.pe else x  # Apply positional encoding if needed

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.projected(x)
        return x, x_len 

def build_encoder(config):
    return ConformerEncoder(config['encoder_params'])