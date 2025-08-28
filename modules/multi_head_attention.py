import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model%n_heads)==0
        self.n_heads = n_heads
        self.d_k = d_model//n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q,k,v: batch_size, seq_len, d_model
        '''
        residual = q
        batch_size, seq_len, d_model = q.size()
        q = self.W_q(q).contiguous().view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(k).contiguous().view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(v).contiguous().view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.d_k) # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            attn.fill_mask(-1e9, mask)

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v).transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        return self.layer_norm(output + residual)