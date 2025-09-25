from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_gate = nn.Linear(config.d_model, config.d_ff)
        self.W_up = nn.Linear(config.d_model, config.d_ff)
        self.W_down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        '''
        input:
            x (b, s, d) or (b*s, d)
        return:
            output (b, s, d) or (b*s, d)
        '''
        output = F.silu(self.W_gate(x))*self.W_up(x)
        output = self.W_down(self.dropout(output))
        return output
    
class MoeMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.top_k = config.num_experts_per_tok
        self.experts = nn.ModuleList(MLP(config) for _ in range(self.num_experts))
        self.gate = nn.Linear(self.d_model, self.num_experts)

    def forward(self, x, mask=None):
        '''
        input:
            x (b, s, d)
        output:
            x (b, s, d)
        '''
        batch_size, seq_len, d_model = x.shape
        
        # 处理掩码：展平并计算有效token的掩码
        if mask is not None:
            # 将掩码从(b, s)转换为(b*s,)，只保留有效token
            flat_mask = mask.diagonal(dim1=2, dim2=3).squeeze().contiguous().view(-1)  # (b*s,)
            valid_indices = torch.where(flat_mask == 1)[0]
            num_valid_tokens = valid_indices.shape[0]
            
            if num_valid_tokens == 0:
                # 没有有效token时返回零损失
                return x, torch.tensor(0.0, device=x.device)
            
            # 只对有效token计算路由
            x_flat = x.view(-1, self.d_model)  # (b*s, d)
            valid_x = x_flat[valid_indices]    # (num_valid, d)
        else:
            # 没有掩码时使用所有token
            x_flat = x.view(-1, self.d_model)
            valid_x = x_flat
            num_valid_tokens = x_flat.shape[0]
            valid_indices = torch.arange(num_valid_tokens, device=x.device)
        
        # 计算路由权重
        router_logits = self.gate(valid_x)  # (num_valid, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # 计算基于有效token的负载均衡损失
        # 1. 计算每个专家的平均分配概率（只考虑有效token）
        expert_probs = routing_weights.mean(dim=0)  # (num_experts,)

        # 2. 理想的均匀分布
        ideal_prob = torch.ones_like(expert_probs) / self.num_experts

        # 3. 计算KL散度损失（添加epsilon防止log(0)）
        balance_loss = F.kl_div(
            (expert_probs + 1e-9).log(),  # 输入是log概率
            ideal_prob,                   # 目标是概率分布
            reduction="sum"         # 适当的归约方式
        )
        
        # 继续处理专家选择和输出计算
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # 初始化输出（只针对有效token）
        valid_output = torch.zeros_like(valid_x)
        for expert_idx, expert in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            output = expert(valid_x[top_x])
            valid_output.index_add_(0, top_x, output * routing_weights[top_x, idx, None])
        
        # 将有效token的输出放回到原始位置
        moe_output = torch.zeros_like(x_flat)
        moe_output[valid_indices] = valid_output
        moe_output = moe_output.reshape(batch_size, seq_len, d_model)
        
        return moe_output, balance_loss




if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from model_config import Config
    model_config = Config()
    mlp = MLP(model_config)
    x = torch.randn((model_config.batch_size, model_config.seq_len, model_config.d_model))
    print(mlp(x))