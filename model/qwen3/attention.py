from torch import nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_attention_heads = config.num_attention_heads # 和 query 保持一致
        self.num_key_value_heads = config.num_key_value_heads
        
        assert(self.d_model % self.num_attention_heads == 0)
        assert(self.num_attention_heads % self.num_key_value_heads == 0)

        self.num_queries_per_kv = self.num_attention_heads//self.num_key_value_heads
        self.d_k = self.d_model // self.num_attention_heads
        self.Wq = nn.Linear(self.d_model, self.d_k * self.num_attention_heads)
        self.Wk = nn.Linear(self.d_model, self.d_k * self.num_key_value_heads)
        self.Wv = nn.Linear(self.d_model, self.d_k * self.num_key_value_heads)
        self.Wo = nn.Linear(self.d_k * self.num_attention_heads, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RoPE(self.d_k, max_seq_len=config.max_seq_len)

    def forward(self, x, mask=None):
        '''
        input:
            x: (batch_size, seq_len, d_model)
            mask: (1, 1, seq_len, seq_len)
        return:
            output: (batch_size, seq_len, d_model)
        '''
        batch_size, seq_len, _ = x.size()
        q = self.rope(self.Wq(x).view(batch_size, seq_len, self.num_attention_heads, self.d_k).transpose(1, 2))
        k = self.rope(self.Wk(x).view(batch_size, seq_len, self.num_key_value_heads, self.d_k).transpose(1, 2)).repeat_interleave(self.num_queries_per_kv,dim=1)
        v = self.Wv(x).view(batch_size, seq_len, self.num_key_value_heads, self.d_k).transpose(1, 2).repeat_interleave(self.num_queries_per_kv,dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        score = F.softmax(attn, dim=-1) # (bs, nheads, seq_len, seq_len)
        output = torch.matmul(self.dropout(score), v)
        output = self.Wo(output.transpose(2,1).contiguous().view(batch_size, seq_len, -1))
        return output

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        half_dim = dim // 2
        theta = 10000 **-(torch.arange(half_dim).float() / half_dim)
        positions = torch.arange(max_seq_len)
        freqs = positions.unsqueeze(1) * theta.unsqueeze(0)
        self.register_buffer("cos_li", freqs.cos())  # (max_seq_len, half_dim)
        self.register_buffer("sin_li", freqs.sin())  # (max_seq_len, half_dim)

    def forward(self, x):
        '''
        input: x(batch_size, nheads, seq_len, dim)
        '''
        batch_size, nheads, seq_len, dim = x.shape
        # if dim != self.dim:
        #     raise ValueError(f"输入维度 {dim} 与RoPE初始化维度 {self.dim} 不匹配")
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入序列长度 {seq_len} 超过最大序列长度 {self.max_seq_len}")
        cos = self.cos_li[None, None, :seq_len, :dim//2]  # (1, 1, seq_len, half_dim)
        sin = self.sin_li[None, None, :seq_len, :dim//2]  # (1, 1, seq_len, half_dim)
        
        # 将输入分成两半，以便与cos/sin的维度匹配
        x1, x2 = x[..., :dim//2], x[..., dim//2:]  # 各为 (B, H, T, D//2)
        
        # 应用旋转公式，确保维度匹配
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)
        
        return x_rotated

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from model_config import Config
    model_config = Config()

    attn = Attention(model_config)
    x = torch.randn(model_config.batch_size, model_config.seq_len, model_config.d_model)
    attn(x)