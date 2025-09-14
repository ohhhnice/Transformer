import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.transformer import Transformer
from utils.set_seed import set_all_seeds
from torch import nn, optim
from utils.train_function import train_epoch, evaluate_epoch
from load_data.translation_data.transformer_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.transformer_data_loader.tokenizer_for_bert import translation_dataloader_huge
from load_data.translation_data.bert_data_loader.get_vocab import Vocab
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from model.bert_base_seq2seq import Bert_Base_Seq2seq
from transformers import BertTokenizer, BertModel
import torch
import json

def main():
    SEED = 5
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2idx_path = "./load_data/translation_data/bert_data_loader/large_vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/bert_data_loader/large_vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/bert_data_loader/tokenizer/'
    model_config_file = './config/bert_config.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'

    # 加载词表
    model_pth="./model/bert"
    encoder_vocab = BertTokenizer.from_pretrained(model_pth)
    encoder_vocab_size = len(encoder_vocab)
    encoder_padding_idx = encoder_vocab.convert_tokens_to_ids("[PAD]")
    decoder_vocab = Vocab(min_freq=50)
    decoder_vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    decoder_padding_idx = decoder_vocab.word2idx["[PAD]"]
    decoder_vocab_size = len(decoder_vocab)
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    

    # 加载模型
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    # encoder_init_pth='./pth/bert/best_transformer_epoch2.pth'
    encoder_init_pth="./model/bert"
    # model.load_state_dict(torch.load('./pth/bert/best_transformer_epoch2.pth', map_location=device))

    model = Bert_Base_Seq2seq(
        encoder_vocab_size=encoder_vocab_size,
        decoder_vocab_size=decoder_vocab_size,
        encoder_num_layers=model_config['num_encoder_layers'],
        decoder_num_layers=model_config["decoder_num_layers"],
        d_model=model_config['d_model'],
        max_len=model_config['max_len'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        encoder_init_pth=encoder_init_pth,
        dropout=model_config['dropout']
    ).to(device)
    model.freeze_encoder_parameters()  # 冻结编码器参数

    # 加载数据
    train_dataloader, test_dataloader= translation_dataloader_huge(
        batch_size = model_config['batch_size'],
        src_max_len=model_config['max_len'],
        tgt_max_len=model_config['max_len'],
        train_json_file = train_data_path,
        val_json_file = valid_data_path,
        encoder_vocab=encoder_vocab, 
        decoder_vocab=decoder_vocab,
        tokenizer=tokenizer)

    # 训练设置
    criterion = nn.CrossEntropyLoss(ignore_index=decoder_padding_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 5
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, encoder_padding_idx, decoder_padding_idx, device)
        valid_loss = evaluate_epoch(model, test_dataloader, criterion, encoder_padding_idx, decoder_padding_idx, device)

        # if valid_loss < best_valid_loss:
        if True:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_transformer_epoch{}.pth'.format(epoch+1))

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {valid_loss:.4f}')


if __name__ == "__main__":
    main()