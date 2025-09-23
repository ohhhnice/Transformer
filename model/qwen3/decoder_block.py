from torch import nn
import torch.nn.functional as F
from .attention import Attention
from .mlp import MLP

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        residual = x
        attn_output = self.attention(self.attention_norm(x), mask)
        attn_output = attn_output + residual

        residual = attn_output
        output = self.mlp(self.mlp_norm(attn_output))
        output = output + residual 
        return output


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from model_config import Config
    model_config = Config()

    decoder_block = DecoderBlock(model_config)
    x = torch.randn(model_config.batch_size, model_config.seq_len, model_config.d_model)
    decoder_block(x)
