import torch.nn as nn
import math

from .encoder_layer import EncoderLayer
from .position_embedding import PositionEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, seq_len=300, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        '''
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        '''
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x