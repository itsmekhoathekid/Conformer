import torch
import torch.nn as nn
from .modules import Linear, Conv2dSubampling, FeedForwardModule, ConvolutionalModule, ResidualConnection
from .attention import MultiHeadedSelfAttentionModule

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_ratio, dropout, kernel_size):
        super(ConformerBlock, self).__init__()
        
        self.ffm1 = FeedForwardModule(d_model, ff_ratio * d_model, dropout, activation="swish")
        self.attention = MultiHeadedSelfAttentionModule(d_model, n_heads, dropout)
        self.conv_module = ConvolutionalModule(d_model, kernel_size, dropout)
        self.ffm2 = FeedForwardModule(d_model, ff_ratio * d_model, dropout, activation="swish")

        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(4)
        ])
    
    def forward(self, x, mask):
        x = self.residual_connections[0](x, self.ffm1, 0.5)
        x = self.residual_connections[1](x, lambda x: self.attention(x, mask), 1.0)
        x = self.residual_connections[2](x, self.conv_module, 1.0)
        x = self.residual_connections[3](x, self.ffm2, 0.5)
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super(ConformerEncoder, self).__init__()
        self.subsampling = Conv2dSubampling(
            in_channels = config["in_channels"],
            out_channels = config["encoder_dim"],
        )
        self.input_projection = nn.Sequential(
            Linear(config["encoder_dim"] * (((config["input_dim"] - 1) // 2 - 1) // 2), config["encoder_dim"]),
            nn.Dropout(p=config["dropout_rate"]),
        )
        
       
        
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=config["encoder_dim"],
                n_heads=config["num_attention_heads"],
                ff_ratio=config["feed_forward_expansion_factor"],
                dropout=config["dropout_rate"],
                kernel_size=config["conv_kernel_size"]
            ) for _ in range(config["num_encoder_layers"])
        ])
       
        self.projected_dim = Linear(config["encoder_dim"], config["output_dim"])
        

    def forward(self, x, x_length, training=True):
        x, x_length = self.subsampling(x, x_length)  # (batch, time', dim)
        x = self.input_projection(x)  # (batch, time', dim)


        mask = self._generate_mask(x_length, x.size(1)) # (batch, time')
        
        for layer in self.layers:
            x = layer(x, mask)
        x = self.projected_dim(x)
        return x, x_length

    def _generate_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        # lengths: (B,) trên đúng device
        device = lengths.device
        # Phòng khi lengths > max_len do làm tròn/stride ở subsampling
        lengths = lengths.clamp_max(max_len)

        # (1, T') so với (B, 1) -> (B, T'), True là PAD
        seq_range = torch.arange(max_len, device=device)
        mask = seq_range.unsqueeze(0).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
        return mask.unsqueeze(1)



def build_encoder(config):
    return ConformerEncoder(config['encoder_params'])