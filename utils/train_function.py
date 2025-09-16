from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch
from tqdm import tqdm
import pdb

def train_epoch(model, data_loader, criterion, optimizer, encoder_pad_idx, decoder_pad_idx, device):
    model.train()
    total_loss = 0
    for zh, en  in tqdm(data_loader, desc="Train Epoch"):
        # en, zh: (batch_size, seq_len)
        zh, en = zh.to(device), en.to(device)
        src_input = zh
        tgt_input, tgt_output = en[:, :-1], en[:, 1:]
        src_mask = get_padding_mask(src_input, pad_idx=encoder_pad_idx).to(device)
        cross_mask = src_mask
        src_mask = src_mask.squeeze(1).squeeze(1)
        tgt_mask = get_tgt_mask(tgt_input, pad_idx=decoder_pad_idx).to(device)
        

        optimizer.zero_grad()
        src_mask = src_mask.long()
        output = model(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)

        loss = criterion(output.view(-1, output.contiguous().size(-1)), tgt_output.contiguous().view(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss/len(data_loader)


def evaluate_epoch(model, data_loader, criterion, encoder_pad_idx, decoder_pad_idx, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for zh, en in tqdm(data_loader, desc='Eval Epoch'):
            zh, en = zh.to(device), en.to(device)
            src_input = zh
            tgt_input, tgt_output = en[:, :-1], en[:, 1:]
            src_mask = get_padding_mask(src_input, pad_idx=encoder_pad_idx).to(device)
            cross_mask = src_mask
            src_mask = src_mask.squeeze(1).squeeze(1)
            tgt_mask = get_tgt_mask(tgt_input, pad_idx=decoder_pad_idx).to(device)
            # cross_mask = get_padding_mask(tgt_input, pad_idx=pad_idx).to(device)
            
            src_mask = src_mask.long()
            output = model(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)

            loss = criterion(output.view(-1, output.contiguous().size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss/len(data_loader)









