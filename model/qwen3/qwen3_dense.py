from torch import nn
from .decoder_block import DecoderBlock

class Qwen3Dense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input, mask=None):
        x = self.word_embedding(input)
        for layer in self.decoder:
            x = layer(x, mask)
        x = self.fc_out(x)
        return x
        