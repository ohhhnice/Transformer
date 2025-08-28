
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import torch
from modules.multi_head_attention import MultiHeadAttention

batch_size, seq_len, d_model = 2, 5, 6
q,k,v = torch.randn((batch_size, seq_len, d_model)), torch.randn((batch_size, seq_len, d_model)), torch.randn((batch_size, seq_len, d_model))
print(q)

attn = MultiHeadAttention(d_model=6, n_heads=3)
attn(q,k,v)

