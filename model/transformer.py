import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, 
                 d_model, n_heads, d_ff, 
                 encoder_num_layers, decoder_num_layers, 
                 encoder_seq_len, decoder_seq_len, 
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = self.get_encoder(encoder_vocab_size, d_model, n_heads, d_ff, encoder_num_layers, encoder_seq_len, dropout)
        self.decoder = self.get_decoder(decoder_vocab_size, d_model, n_heads, d_ff, decoder_num_layers, decoder_seq_len, dropout)

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
        cross_mask = src_mask
        output = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
        return output
    
    def get_encoder(self, encoder_vocab_size, d_model, n_heads, d_ff, encoder_num_layers, encoder_seq_len, dropout=0.1):
        encoder = Encoder(vocab_size = encoder_vocab_size, 
                          d_model=d_model, 
                          n_heads=n_heads, 
                          d_ff = d_ff, 
                          num_layers = encoder_num_layers, 
                          seq_len = encoder_seq_len, 
                          dropout = dropout)
        return encoder


    def get_decoder(self, decoder_vocab_size, d_model, n_heads, d_ff, decoder_num_layers, decoder_seq_len, dropout=0.1):
        decoder = Decoder(vocab_size = decoder_vocab_size, 
                          d_model = d_model, 
                          n_heads = n_heads, 
                          d_ff = d_ff, 
                          num_layers = decoder_num_layers, 
                          seq_len = decoder_seq_len, 
                          dropout = dropout)
        return decoder























