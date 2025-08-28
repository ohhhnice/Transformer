import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        x: (batch_size, seq_len, d_model)
        mask: (seq_len, seq_len)
        output: (batch_size, seq_len, d_model)
        '''
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))
        return x
        