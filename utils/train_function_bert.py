from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch
from tqdm import tqdm

def train_epoch(model, data_loader, mlm_criterion, nsp_criterion, optimizer, pad_idx, device):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Train Epoch"):
        # en, zh: (batch_size, seq_len)
        merged_tokens_ids, seg_ids, attn_mask, mask_labels, nsp_label = data["merged_tokens_ids"].to(device), \
            data["seg_ids"].to(device), data["attn_mask"].to(device), data["mlm_labels"].to(device), data["nsp_label"].to(device)

        optimizer.zero_grad()
        mlm_output, nsp_output = model(x=merged_tokens_ids, segment_ids=seg_ids, attn_mask=attn_mask)

        mlm_loss = mlm_criterion(mlm_output.view(-1, mlm_output.contiguous().size(-1)), mask_labels.contiguous().view(-1))
        nsp_loss = nsp_criterion(nsp_output, nsp_label.contiguous().view(-1))
        loss = mlm_loss + 0.1*nsp_loss
        
        loss.backward()
        optimizer.step()

        right_mlm = torch.sum((torch.argmax(mlm_output.view(-1, mlm_output.contiguous().size(-1)), dim=-1)==mask_labels.contiguous().view(-1)) & (mask_labels.contiguous().view(-1)!=0))
        all_mlm = torch.sum(mask_labels.contiguous().view(-1)!=0)
        with open("./result_train", "a", encoding="utf-8") as f:
            f.write(f"batch_size: {len(nsp_label)}\tmlm_task right: {right_mlm}/{all_mlm}\tnsp_right: {torch.sum(nsp_label.reshape(-1,)==torch.argmax(nsp_output, dim=-1))}/{len(nsp_label)}\n")
        # print(f"batch_size: {len(nsp_label)}")
        # print(f"mlm_task right: {right_mlm}/{all_mlm}")
        # print(f"nsp_right: {torch.sum(nsp_label.reshape(-1,)==torch.argmax(nsp_output, dim=-1))}/{len(nsp_label)}")
        total_loss += loss.item()

    return total_loss/len(data_loader)


def evaluate_epoch(model, data_loader, mlm_criterion, nsp_criterion, pad_idx, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data  in tqdm(data_loader, desc="Eval Epoch"):
            merged_tokens_ids, seg_ids, attn_mask, mask_labels, nsp_label = data["merged_tokens_ids"].to(device), \
                data["seg_ids"].to(device), data["attn_mask"].to(device), data["mlm_labels"].to(device), data["nsp_label"].to(device)
            mlm_output, nsp_output = model(x=merged_tokens_ids, segment_ids=seg_ids, attn_mask=attn_mask)

            mlm_loss = mlm_criterion(mlm_output.view(-1, mlm_output.contiguous().size(-1)), mask_labels.contiguous().view(-1))
            nsp_loss = nsp_criterion(nsp_output, nsp_label.contiguous().view(-1))
            loss = mlm_loss + 0.1*nsp_loss
            total_loss += loss.item()

            right_mlm = torch.sum((torch.argmax(mlm_output.view(-1, mlm_output.contiguous().size(-1)), dim=-1)==mask_labels.contiguous().view(-1)) & (mask_labels.contiguous().view(-1)!=0))
            all_mlm = torch.sum(mask_labels.contiguous().view(-1)!=0)
            with open("./result_val", "a", encoding="utf-8") as f:
                f.write(f"batch_size: {len(nsp_label)}\tmlm_task right: {right_mlm}/{all_mlm}\tnsp_right: {torch.sum(nsp_label.reshape(-1,)==torch.argmax(nsp_output, dim=-1))}/{len(nsp_label)}\n")

    return total_loss/len(data_loader)









