import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):

    """Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Attention Params
        self.num_heads = num_heads # H
        self.dim_model = dim_model # D
        self.dim_head = dim_model // num_heads # d

        # Linear Layers
        self.query_layer = nn.Linear(self.dim_model, self.dim_model)
        self.key_layer = nn.Linear(self.dim_model, self.dim_model)
        self.value_layer = nn.Linear(self.dim_model, self.dim_model)
        self.output_layer = nn.Linear(self.dim_model, self.dim_model)

    def forward(self, Q, K, V, mask=None):

        """Scaled Dot-Product Multi-Head Attention

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, T)

        """

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T, T)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask * -1e9)

        # Att weights (B, H, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach()

    def pad(self, Q, K, V, mask, chunk_size):

        # Compute Overflows
        overflow_Q = Q.size(1) % chunk_size
        overflow_KV = K.size(1) % chunk_size
        
        padding_Q = chunk_size - overflow_Q if overflow_Q else 0
        padding_KV = chunk_size - overflow_KV if overflow_KV else 0

        batch_size, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0)

        # Update Padding Mask
        if mask is not None:

            # (B, 1, 1, T) -> (B, 1, 1, T + P) 
            if mask.size(2) == 1:
                mask = F.pad(mask, pad=(0, padding_KV), value=1)
            # (B, 1, T, T) -> (B, 1, T + P, T + P)
            else:
                mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=1)

        elif padding_KV:

            # None -> (B, 1, 1, T + P) 
            mask = F.pad(Q.new_zeros(batch_size, 1, 1, seq_len_KV), pad=(0, padding_KV), value=1)

        return Q, K, V, mask, padding_Q

class RelativeSinusoidalPositionalEncoding(nn.Module):
    
    """
        Relative Sinusoidal Positional Encoding

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - 1
    """

    def __init__(self, max_len, dim_model, causal=False):
        super(RelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - 1, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len-1, end=0, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:,:self.max_len]

        # Full Context
        else:

            # (B, Th + 2*T-1, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len - 1  + seq_len]
            
            # (B, 2*Tmax-1, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)

class RelPosMultiHeadSelfAttention(MultiHeadAttention):

    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References: 
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(self, dim_model, num_heads, causal, max_pos_encoding):
        super(RelPosMultiHeadSelfAttention, self).__init__(dim_model, num_heads)

        # Position Embedding Layer
        self.pos_layer = nn.Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
        self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
        torch.nn.init.xavier_uniform_(self.u.reshape(self.num_heads, self.dim_head)) # glorot uniform
        torch.nn.init.xavier_uniform_(self.v.reshape(self.num_heads, self.dim_head)) # glorot uniform

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = RelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.causal)

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References: 
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
            
            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores

    def forward(self, Q, K, V, mask=None, hidden=None):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Batch size B
        batch_size = Q.size(0)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        # Hidden State Provided
        if hidden:
            K = torch.cat([hidden["K"], K], dim=1)
            V = torch.cat([hidden["V"], V], dim=1)

        # Update Hidden State
        hidden = {"K": K.detach(), "V": V.detach()}

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T, Th + T)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # print(att_scores.shape)
        # print(mask.shape)
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            att_scores += (mask * -1e9)

        # Att weights (B, H, T, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.output_layer(O)

        return O, att_w.detach(), hidden


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # print("Mask shape:", mask.shape)  # [B, T]
            # print("attention_scores shape:", attention_scores.shape)  # [B, h, T    , T]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        att_w = None
        _ = None
        return self.w_o(x), att_w, _

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, num_heads, dim_model, dropout=0.1, max_pos_encoding=512, attention_type = 'mha'):
        super(MultiHeadSelfAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dropout = dropout

        # Multi-Head Attention Layer
        if attention_type == 'mha':
            self.mha = MultiHeadAttentionBlock(dim_model, num_heads, dropout = dropout)
        else:
            self.mha = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal=True, max_pos_encoding=max_pos_encoding)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x, mask=None, hidden=None): 
        residueal = x
        x = self.layer_norm(x)
        x, att_w, hidden = self.mha(x, x, x, mask=mask)
        x = self.dropout_layer(x)
        x = x + residueal
        return x, att_w, hidden

# https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py
# https://github.com/burchim/EfficientConformer/blob/master/models/encoders.py 