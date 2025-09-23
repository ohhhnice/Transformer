from torch import nn
import torch
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert(d_model % n_heads == 0)
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        half_dim = dim // 2
        theta = 10000 **-(torch.arange(half_dim).float() / half_dim)
        positions = torch.arange(max_seq_len)
        freqs = positions.unsqueeze(1) * theta.unsqueeze(0)
        self.register_buffer("cos_li", freqs.cos().unsqueeze(0))  # (1, max_seq_len, half_dim)
        self.register_buffer("sin_li", freqs.sin().unsqueeze(0))  # (1, max_seq_len, half_dim)

    def forward(self, x):
        '''
        input: x(batch_size, seq_len, dim)
        '''
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"输入维度 {dim} 与RoPE初始化维度 {self.dim} 不匹配")
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入序列长度 {seq_len} 超过最大序列长度 {self.max_seq_len}")
        cos = self.cos_li[:, :seq_len, :]  # (1, seq_len, half_dim)
        sin = self.sin_li[:, :seq_len, :]  # (1, seq_len, half_dim)
        
        # 将输入分成两半，以便与cos/sin的维度匹配
        x1, x2 = x[..., :dim//2], x[..., dim//2:]  # 各为 (batch_size, seq_len, half_dim)
        
        # 应用旋转公式，确保维度匹配
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return x_rotated

if __name__ == "__main__":
    rope = RoPE(6)
    x = torch.randn(2, 5, 6)
    rope(x)