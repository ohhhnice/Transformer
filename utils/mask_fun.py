import torch
def get_padding_mask(src, pad_idx=0):
    # src: (batch_size, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
    return src_mask

def get_tgt_mask(tgt, pad_idx=0):
    # tgt: (batch_size, tgt_len)
    tgt_pad_mask = get_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tri_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = tgt.device)).bool()
    return tgt_pad_mask & tri_mask