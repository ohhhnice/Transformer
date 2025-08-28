import torch.nn as nn
import math

from .decoder_layer import DecoderLayer
from .position_embedding import PositionEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, seq_len=300, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        '''
        x: (batch_size, seq_len, d_model)
        enc_output: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, vocab_size)
        '''
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
        x = self.fc_out(x)
        return x