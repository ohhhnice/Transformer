import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import torch
from utils.set_seed import set_all_seeds
from model.bert import Bert_For_Decoder
from load_data.translation_data.bert_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.bert_data_loader.get_vocab import Vocab
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from utils.train_function_bert import train_epoch, evaluate_epoch
import random
import torch.nn as nn
from modules.decoder import Decoder

class Bert_Base_Seq2seq(nn.Module):
    def __init__(self, vocab_size, encoder_num_layers, decoder_num_layers, d_model, max_len, 
                 n_heads, d_ff, encoder_init_pth, dropout=0.1):
        super(Bert_Base_Seq2seq, self).__init__()
        self.encoder = self.get_encoder(vocab_size=vocab_size,
                                        encoder_num_layers=encoder_num_layers,
                                        d_model=d_model,
                                        max_len=max_len,
                                        n_heads=n_heads,
                                        d_ff=d_ff,
                                        encoder_init_pth=encoder_init_pth,
                                        dropout=dropout)
        self.decoder = self.get_decoder(decoder_vocab_size=vocab_size, 
                                        d_model=d_model, 
                                        n_heads=n_heads,
                                        d_ff=d_ff, 
                                        decoder_num_layers=decoder_num_layers, 
                                        decoder_seq_len=max_len, 
                                        dropout=dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        seg_ids = torch.zeros_like(src)
        enc_output = self.encoder(x=src, segment_ids=seg_ids, attn_mask=src_mask)
        output = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
        return output

    def get_encoder(self,vocab_size,encoder_num_layers,d_model,max_len,n_heads,d_ff,encoder_init_pth,dropout):
        encoder = Bert_For_Decoder(vocab_size=vocab_size,
                 encoder_num_layers=encoder_num_layers,
                 d_model=d_model,
                 max_len=max_len,
                 n_heads=n_heads,
                 d_ff=d_ff,
                 dropout=dropout)
        encoder.load_state_dict(torch.load(encoder_init_pth))
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
    
    def freeze_encoder_parameters(self):
        """冻结编码器所有参数，使其不参与训练更新"""
        for param in self.encoder.parameters():
            param.requires_grad = False  # 核心：禁用梯度计算
        
    def unfreeze_encoder_parameters(self):
        """可选：解冻编码器参数，用于后续微调"""
        for param in self.encoder.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    SEED = 5
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2idx_path = "./load_data/translation_data/bert_data_loader/large_vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/bert_data_loader/large_vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/bert_data_loader/tokenizer/'
    model_config_file = './config/bert_config.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train_test.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'

    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载模型
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    encoder_init_pth='./pth/bert/best_transformer_epoch2.pth'
    # model.load_state_dict(torch.load('./pth/bert/best_transformer_epoch2.pth', map_location=device))
    model = Bert_Base_Seq2seq(
        vocab_size=len(vocab),
        encoder_num_layers=model_config['num_encoder_layers'],
        decoder_num_layers=model_config["decoder_num_layers"],
        d_model=model_config['d_model'],
        max_len=model_config['max_len'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        encoder_init_pth=encoder_init_pth,
        dropout=model_config['dropout']
    )

    src = torch.tensor([1 for _ in range(10)]).unsqueeze(0)
    tgt = torch.tensor([1 for _ in range(10)]).unsqueeze(0)
    model(src, tgt)