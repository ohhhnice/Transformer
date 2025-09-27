import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import torch
from utils.set_seed import set_all_seeds
from model.bert import Bert
from load_data.translation_data.qwen3_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.qwen3_data_loader.get_vocab import Vocab
from load_data.translation_data.qwen3_data_loader.get_tokenizer import get_multilingual_tokenizer
from utils.train_function_qwen3 import train_epoch, evaluate_epoch

from model.qwen3.model_config import Config
from model.qwen3.qwen3 import Qwen3
from model.qwen3.mlp import MoeMlp




def main():
    SEED = 5
    num_epochs = 3
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train_test.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'
    word2idx_path = "./load_data/translation_data/qwen3_data_loader/vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/qwen3_data_loader/vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/qwen3_data_loader/tokenizer/'


    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载数据
    model_config = Config()
    train_dataloader, valid_dataloader = translation_dataloader(
        batch_size=model_config.batch_size, 
        max_len=model_config.max_seq_len, 
        tokenizer=tokenizer,    
        vocab=vocab, 
        padding_idx=padding_idx, 
        train_json_file=train_data_path,
        valid_json_file=valid_data_path
    )

    # 加载模型设置
    model = Qwen3(model_config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, padding_idx, device)
        valid_loss = evaluate_epoch(model, valid_dataloader, criterion, padding_idx, device)

        # if valid_loss < best_valid_loss:
        if True:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './best_transformer_epoch{}.pth'.format(epoch+1))

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {valid_loss:.4f}')
        with open("result", "a", encoding="utf-8") as f:
            f.write(f"Epoch: {epoch+1}\ttrain_loss: {train_loss}\tval_loss: {valid_loss}\n")

    
    


if __name__ == "__main__":
    main()