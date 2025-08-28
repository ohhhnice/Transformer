import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        '''
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        src_mask: (src_seq_len, src_seq_len)
        tgt_mask: (tgt_seq_len, tgt_seq_len)
        cross_mask: (tgt_seq_len, src_seq_len)
        output: (batch_size, tgt_seq_len, vocab_size)
        '''
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
        return output






















