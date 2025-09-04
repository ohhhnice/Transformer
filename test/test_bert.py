import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import json
from load_data.translation_data.bert_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from load_data.translation_data.bert_data_loader.get_vocab import Vocab


if __name__=="__main__":
    tokenizer = get_multilingual_tokenizer()
    vocab = Vocab(min_freq=2)
    train_dataloader, test_dataloader = translation_dataloader(
        batch_size=32, max_len=50, tokenizer=tokenizer, vocab=vocab, json_file="./load_data/translation_data/translation2019zh/translation2019zh_valid.json"
    )
    for batch in train_dataloader:
        print(batch)
        break