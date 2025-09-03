def get_mask(input_ids, pad_token_id=0):
    """
    获取注意力掩码（Attention Mask）
    input_ids: Tensor, shape (batch_size, seq_len)
    pad_token_id: int, padding token的ID
    return: Tensor, shape (batch_size, 1, 1, seq_len)
    """
    # 创建掩码，padding位置为0，其他位置为1
    attn_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    # 将bool类型转换为float类型
    attn_mask = attn_mask.float()
    # 将padding位置设为一个很小的值，防止其在softmax中被关注
    attn_mask = (1.0 - attn_mask) * -10000.0
    return attn_mask