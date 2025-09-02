import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import torch
from utils.set_seed import set_all_seeds
from model.bert import Bert


def main():
    SEED = 5
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 加载模型设置
    model_config_file = './config/bert_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    batch_size = 2
    seq_len = 10
    vocab_size = 30
    
    model = Bert(vocab_size=vocab_size,
                 d_model=model_config['d_model'],
                 max_len=model_config['max_len'],
                 dropout=model_config['dropout']).to(device)
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)  # (batch_size, seq_len)
    segment_ids = torch.randint(0, 2, (batch_size, seq_len)).to(device)  # (batch_size, seq_len)
    model(x=x, segment_ids=segment_ids)


if __name__ == "__main__":
    main()