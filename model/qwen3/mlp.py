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

    def forward(self, x):
        '''
        input:
            x (b, s, d)
        output:
            x (b, s, d)
        '''
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, self.d_model) # (b*s, d)
        router_logits = self.gate(x) # (b*s, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float) # 计算权重
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k) # 计算每个token归属哪个experts
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        moe_output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue # 没有token被分配给此专家则跳过
            output = expert(x[top_x])
            moe_output.index_add_(0, top_x, output*routing_weights[top_x, idx, None])

        moe_output = moe_output.reshape(batch_size, seq_len, d_model)
        return moe_output




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