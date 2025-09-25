from torch import nn
from .decoder_block import DecoderBlock

class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, mask=None):
        '''
        x (b, s)
        mask (b, s, s)
        '''
        balance_loss = 0.0
        x = self.word_embedding(x)
        for layer in self.decoder:
            x, balance_loss_i = layer(x, mask)
            balance_loss = balance_loss + balance_loss_i

        x = self.fc_out(x)
        return x, balance_loss
        