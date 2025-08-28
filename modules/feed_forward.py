import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.Linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.GELU()

    def forward(self, x):
        '''
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        '''
        x = self.Linear1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        return x
        
        