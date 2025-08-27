from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch

def train_epoch(model, data_loader, criterion, optimizer, pad_idx, device):
    model.train()
    total_loss = 0
    for en, zh in data_loader:
        # en, zh: (batch_size, seq_len)
        en, zh = en.to(device), zh.to(device)
        src_input = zh
        tgt_input, tgt_output = en[:, :-1], en[:, 1:]
        src_mask = get_padding_mask(src_input, pad_idx=pad_idx).to(device)
        tgt_mask = get_tgt_mask(tgt_input, pad_idx=pad_idx).to(device)

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
        for en, zh in data_loader:
            en, zh = en.to(device), zh.to(device)
            src_input = zh
            tgt_input, tgt_output = en[:, :-1], en[:, 1:]
            src_mask = get_padding_mask(src_input, pad_idx=pad_idx).to(device)
            tgt_mask = get_tgt_mask(tgt_input, pad_idx=pad_idx).to(device)

            output = model(src=src_input, tgt=tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)

            loss = criterion(output.view(-1, output.contiguous().size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

    return total_loss/len(data_loader)









