
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# from model.transformer import Transformer
from utils.set_seed import set_all_seeds
from torch import nn, optim
from load_data.translation_data.get_dataloader import translation_dataloader
import torch
import json

import torch
from tqdm import tqdm

import torch
def get_padding_mask(src, pad_idx=0):
    # src: (batch_size, src_len)
    key_padding_mask = torch.zeros(src.size())
    key_padding_mask[src == pad_idx] = -torch.inf
    return key_padding_mask

def get_tgt_mask(tgt, pad_idx=0):
    # tgt: (batch_size, tgt_len)
    tgt_pad_mask = get_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tri_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = tgt.device)).bool()
    return tgt_pad_mask & tri_mask

def train_epoch(model, data_loader, criterion, optimizer, pad_idx, device):
    model.train()
    total_loss = 0
    for zh, en  in tqdm(data_loader, desc="Train Epoch"):
        # en, zh: (batch_size, seq_len)
        zh, en = zh.to(device), en.to(device)
        src_input = zh
        tgt_input, tgt_output = en[:, :-1], en[:, 1:]
        src_mask = get_padding_mask(src_input, pad_idx=pad_idx).to(device)
        tgt_mask = get_padding_mask(tgt_input, pad_idx=pad_idx).to(device)

        optimizer.zero_grad()
        output = model(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

        loss = criterion(output.view(-1, output.contiguous().size(-1)), tgt_output.contiguous().view(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(data_loader)


def evaluate_epoch(model, data_loader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for zh, en in tqdm(data_loader, desc='Eval Epoch'):
            zh, en = zh.to(device), en.to(device)
            src_input = zh
            tgt_input, tgt_output = en[:, :-1], en[:, 1:]
            src_mask = get_padding_mask(src_input, pad_idx=pad_idx).to(device)
            tgt_mask = get_padding_mask(tgt_input, pad_idx=pad_idx).to(device)

            output = model(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

            loss = criterion(output.view(-1, output.contiguous().size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss/len(data_loader)



class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, d_model=512, n_heads=8, encoder_num_layers=6, decoder_num_layers=6, d_ff=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=encoder_num_layers,
        num_decoder_layers=decoder_num_layers, dim_feedforward=d_ff,
        dropout=dropout,
        batch_first=True)
        self.src_embed = nn.Embedding(encoder_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(decoder_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        tgt_mask_causal = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(tgt.device)
        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)
        output = self.transformer(src=src, 
                                  tgt=tgt,
                                  tgt_mask = tgt_mask_causal,
                                  src_key_padding_mask = src_mask,
                                  tgt_key_padding_mask = tgt_mask)
        return self.fc_out(output)


def main():
    SEED = 5
    batch_size = 32
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 加载模型设置
    model_config_file = './config/transformer_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    # 加载数据
    train_dataloader, test_dataloader, src_vocab, tgt_vocab, _, _ = translation_dataloader(
        batch_size = batch_size,
        max_len=model_config['seq_len'], seed=SEED,
        json_file = './load_data/translation_data/translation2019zh/translation2019zh_valid.json')
    padding_idx = src_vocab.word2idx["<PAD>"]

    # load model
    model = Transformer(encoder_vocab_size = len(src_vocab), 
                        decoder_vocab_size = len(tgt_vocab), 
                        d_model = model_config['d_model'], 
                        n_heads = model_config['n_heads'], 
                        d_ff = model_config['d_ff'], 
                        encoder_num_layers = model_config['num_encoder_layers'], 
                        decoder_num_layers = model_config['num_decoder_layers'], 
                        # encoder_seq_len = model_config['seq_len'], 
                        # decoder_seq_len = model_config['seq_len'], 
                        dropout = model_config['dropout']).to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
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


















