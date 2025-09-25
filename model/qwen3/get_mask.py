import torch
def generate_qwen3_mask(input_ids: torch.Tensor, pad_token_id: int = 1):
    '''
    需要掩码的位置是0，非掩码位置是1
    '''
    base_mask = (input_ids != pad_token_id).float()
    
    # 增加维度以适应多头注意力的需求
    # 形状变为: [batch_size, 1, seq_len]
    expanded_mask = base_mask.unsqueeze(1)
    
    # 创建因果掩码（上三角矩阵），确保模型只能关注当前位置及之前的位置
    # 形状: [1, seq_len, seq_len]
    seq_len = input_ids.size(1)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device)).unsqueeze(0)
    
    # 结合基础掩码和因果掩码，得到最终的注意力掩码
    # 形状: [batch_size, 1, seq_len, seq_len]
    attention_mask = expanded_mask * causal_mask
    
    return attention_mask