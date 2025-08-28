import torch.nn as nn
import torch
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, seq_len=300, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp( torch.arange(0, d_model, 2) * (-math.log(10000) / d_model) )
        pe = torch.zeros((seq_len, d_model))
        pe[:,0::2] = torch.sin(position * div)
        pe[:,1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        '''
        return self.dropout(x + self.pe[:x.size(1),:])