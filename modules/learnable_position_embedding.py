import torch.nn as nn
import torch

class LearnablePositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(LearnablePositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        # nn.init.xavier_uniform_(self.pos_embedding.weight)

    def forward(self, x):
        '''
        x: Tensor, shape (batch_size, seq_len, d_model)
        return: Tensor, shape (batch_size, seq_len, d_model)
        '''
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0) # (1, seq_len)
        pos_embedding = self.pos_embedding(pos)
        return self.dropout(x + pos_embedding)
