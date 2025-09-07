import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import torch
from utils.set_seed import set_all_seeds
from model.bert import Bert
from load_data.translation_data.bert_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.bert_data_loader.get_vocab import Vocab
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from utils.train_function_bert import train_epoch, evaluate_epoch


def main():
    SEED = 5
    num_epochs = 3000
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train_test.json'
    # train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'
    word2idx_path = "./load_data/translation_data/bert_data_loader/vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/bert_data_loader/vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/bert_data_loader/tokenizer/'
    model_config_file = './config/bert_config.json'

    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载数据
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    train_dataloader, valid_dataloader = translation_dataloader(
        batch_size=model_config["batch_size"], 
        max_len=model_config["max_len"], 
        tokenizer=tokenizer,    
        vocab=vocab, 
        padding_idx=padding_idx, 
        train_json_file=train_data_path,
        valid_json_file=valid_data_path
    )

    # 加载模型设置
    model = Bert(vocab_size=len(vocab),
                 encoder_num_layers=model_config['num_encoder_layers'],
                 d_model=model_config['d_model'],
                 max_len=model_config['max_len'],
                 n_heads=model_config['n_heads'],
                 d_ff=model_config['d_ff'],
                 dropout=model_config['dropout']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    mlm_criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
    nsp_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, mlm_criterion, nsp_criterion, optimizer, padding_idx, device)
        valid_loss = evaluate_epoch(model, valid_dataloader, mlm_criterion, nsp_criterion, padding_idx, device)

        # if valid_loss < best_valid_loss:
        if True:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './tmp_pth/best_transformer_epoch{}.pth'.format(epoch+1))

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {valid_loss:.4f}')
        with open("result", "a", encoding="utf-8") as f:
            f.write(f"Epoch: {epoch+1}\ttrain_loss: {train_loss}\tval_loss: {valid_loss}\n")

    
    


if __name__ == "__main__":
    main()