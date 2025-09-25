from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch
from tqdm import tqdm

def train_epoch(model, data_loader, criterion, optimizer, pad_idx, device):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Train Epoch"):
        input_ids, label_ids, attn_mask = data["input_ids"].to(device), data["label_ids"].to(device), data["attn_mask"].to(device)
        optimizer.zero_grad()
        output= model(x=input_ids, mask=attn_mask)
        loss = criterion(output.view(-1, output.contiguous().size(-1)), label_ids.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(data_loader)


def evaluate_epoch(model, data_loader, criterion, pad_idx, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data  in tqdm(data_loader, desc="Eval Epoch"):
            input_ids, label_ids, attn_mask = data["input_ids"].to(device), data["label_ids"].to(device), data["attn_mask"].to(device)
            output= model(x=input_ids, attn_mask=attn_mask)

            loss = criterion(output.view(-1, output.contiguous().size(-1)), label_ids.contiguous().view(-1))
            total_loss += loss.item()
    return total_loss/len(data_loader)









