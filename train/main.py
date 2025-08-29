
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.transformer import Transformer
from utils.set_seed import set_all_seeds
from torch import nn, optim
from utils.train_function import train_epoch, evaluate_epoch
from load_data.translation_data.get_dataloader import translation_dataloader
import torch
import json

def main():
    SEED = 10
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 加载模型设置
    model_config_file = './config/transformer_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    # 加载数据
    train_dataloader, test_dataloader, src_vocab, tgt_vocab, _, _ = translation_dataloader(
        max_len=model_config['seq_len'], seed=SEED)
    padding_idx = src_vocab.word2idx["<PAD>"]

    # load model
    model = Transformer(encoder_vocab_size = len(src_vocab), 
                        decoder_vocab_size = len(tgt_vocab), 
                        d_model = model_config['d_model'], 
                        n_heads = model_config['n_heads'], 
                        d_ff = model_config['d_ff'], 
                        encoder_num_layers = model_config['num_encoder_layers'], 
                        decoder_num_layers = model_config['num_decoder_layers'], 
                        encoder_seq_len = model_config['seq_len'], 
                        decoder_seq_len = model_config['seq_len'], 
                        dropout = model_config['dropout']).to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 20
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, padding_idx, device)
        valid_loss = evaluate_epoch(model, test_dataloader, criterion, padding_idx, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_transformer.pth')

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {valid_loss:.4f}')


if __name__ == "__main__":
    main()


















