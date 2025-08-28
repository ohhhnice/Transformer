import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        '''
        x: (batch_size, seq_len, d_model)
        enc_output: (batch_size, seq_len, d_model)
        outupt: (batch_size, seq_len, d_model)
        '''
        attn_output = self.self_attn(x, x, x, self_mask)
        x = self.layer_norm1(x + self.dropout1(attn_output)) 
        attn_output = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.layer_norm2(x + self.dropout2(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm3(x + self.dropout3(ffn_output))
        return x
        

        