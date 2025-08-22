
import torch
import torch.nn as nn
import math

class Swish(torch.nn.Module):
    """The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf

    given input x. Swish(x) = x / (1 + exp(beta * x))

    Arguments
    ---------
    beta: float
        Beta value.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        """Returns the Swished input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        The swished output.
        """
        if self.beta != 1:  # slow path
            x = x * self.beta

        return self.silu(x)

class ConvolutionModule(nn.Module):
    def __init__(self, dim_model, dim_expand, kernel_size, Pdrop, stride, padding, dilation = 1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.conv1d = nn.Conv1d(dim_model, dim_expand * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride


        if padding == 'causal':
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:  # 'valid'
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2
    
        self.conv1d_depthwise = nn.Conv1d(
            in_channels=dim_expand,
            out_channels=dim_expand,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,       # <— key change
            groups=dim_expand
        )

        self.bn = nn.BatchNorm1d(dim_expand)
        self.swish = Swish()
        self.conv1d_2 = nn.Conv1d(dim_expand, dim_expand, kernel_size=1)
        self.dropout = nn.Dropout(Pdrop)

    def forward(self, x):
        # x: (B, T, D)
        residual = x 
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)

        x = self.conv1d(x)     # (B, 2*E, T)
        x = self.glu(x)        # (B, E, T)


        x = self.conv1d_depthwise(x)  # (B, E, T')  — T' now matches for 'same', or grows with causal pad

        x = self.bn(x)
        x = self.swish(x)
        x = self.conv1d_2(x)
        x = x.transpose(1, 2)  # (B, T', E)
        x = self.dropout(x)
        x = x + residual  # (B, T', E)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)

        x = x + pe
        return x


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)

        def forward(self, x, sublayer):
            return self.norm(x + self.dropout(sublayer(x)))

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float, act) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-6),
            nn.Linear(d_model, d_ff),
            Swish() if act == "swish" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layers(x) + residual
        return x


class ConvolutionResidual(nn.Module):
    def __init__(self, dim_model, dim_expand, kernel_size, stride):
        super(ConvolutionResidual, self).__init__()
        self.code = 0
        if dim_model != dim_expand:
            self.layer = nn.Conv1d(
                in_channels=dim_model,
                out_channels=dim_expand,
                kernel_size=1,
                stride=stride,
            )
        elif stride > 1:
            self.layer = nn.MaxPool1d(kernel_size=stride, stride=stride)
    
        else:
            self.code = 1
            self.layer = nn.Identity()
    
    def forward(self, x):
        if self.code == 0:
            x = x.transpose(1, 2)
            x = self.layer(x)
            x = x.transpose(1, 2)
        else:
            x = self.layer(x)
        return x

class AttentionResidual(nn.Module):
    def __init__(self, att_stride):
        super(AttentionResidual, self).__init__()
        self.att_stride = att_stride
        if att_stride > 1:
            self.layer = nn.MaxPool1d(kernel_size=att_stride, stride=att_stride)
        else:
            self.layer = nn.Identity()
    
    def forward(self, x):
        if self.att_stride > 1:
            x = x.transpose(1, 2)
            x = self.layer(x)
            x = x.transpose(1, 2)
        else:
            x = self.layer(x)
        return x

class Conv2dSubsampling(nn.Module):

    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(Conv2dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Conv 2D Subsampling Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, stride=2, padding=(kernel_size - 1) // 2), 
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1

        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.permute(0, 2, 1, 3)  # (B, D, C, T)
        x = x.reshape(batch_size, subsampled_dim, channels *subsampled_length)

        return x, x_len

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)