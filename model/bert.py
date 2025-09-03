import torch.nn as nn
from modules.learnable_position_embedding import LearnablePositionEmbedding
from modules.encoder_layer import EncoderLayer
import torch


class Bert(nn.Module):
    def __init__(self, vocab_size, encoder_num_layers, d_model, max_len, n_heads, d_ff, dropout=0.1):
        super(Bert, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = LearnablePositionEmbedding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.segment_embedding = nn.Embedding(2, d_model)  # 2个句子
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.Encode_layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_heads = n_heads, d_ff = d_ff, dropout=dropout) for _ in range(encoder_num_layers)])
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)
        self._init_weights()
        self.mlm_head.weight = self.embedding.weight
        

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        

    def forward(self, x, segment_ids=None, attn_mask=None):
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
        x = self.dropout(self.layer_norm(x))

        for layer in self.Encode_layers:
            x = layer(x, attn_mask)
        mlm_output = self.mlm_head(x)
        nsp_output = self.nsp_head(x[:,0,:])
        return mlm_output, nsp_output