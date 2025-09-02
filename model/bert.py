import torch.nn as nn
from modules.learnable_position_embedding import LearnablePositionEmbedding
import torch


class Bert(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super(Bert, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = LearnablePositionEmbedding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.segment_embedding = nn.Embedding(2, d_model)  # 2个句子

    def forward(self, x, segment_ids=None):
        '''
        x: Tensor, shape (batch_size, seq_len)
        segment_ids: Tensor, shape (batch_size, seq_len), optional
        return: Tensor, shape (batch_size, seq_len, d_model)
        '''
        if segment_ids is None:
            segment_ids = torch.zeros_like(x)
        x = self.embedding(x)
        x = x + self.segment_embedding(segment_ids)
        x = self.position_embedding(x)
        return x