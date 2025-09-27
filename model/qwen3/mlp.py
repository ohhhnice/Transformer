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
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim) # 重塑为 (batch_size * seq_len, hidden_dim)

        # 计算路由逻辑值
        router_logits = self.gate(x) # (batch_size * seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # 计算softmax得到路由权重

        # 选取top-k专家及其权重
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # 对top-k权重进行归一化

        routing_weights = routing_weights.to(x.dtype) # 转换回原始数据类型
        final_x = torch.zeros(
            (batch_size * seq_len, hidden_dim), 
            dtype=x.dtype, 
            device=x.device
        )

        # 生成专家掩码：用于标识哪些token被分配给了哪些专家
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 遍历所有专家，只计算被分配到的token
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx]) # 找到属于当前专家的token索引

            if top_x.shape[0] == 0:
                continue # 没有token被分配给此专家则跳过

            current_state = x[None, top_x].reshape(-1, hidden_dim) # 获取属于当前专家的token隐藏状态
            current_x = expert_layer(current_state) # 专家前向传播

            # 用路由权重加权专家输出，并通过index_add_累加到最终输出中
            current_x = current_x * routing_weights[top_x, idx, None]
            final_x.index_add_(0, top_x, current_x.to(x.dtype))

        final_x = final_x.reshape(batch_size, seq_len, hidden_dim)
        return final_x, router_logits




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