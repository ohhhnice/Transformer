import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BalancedMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, expert_dim, capacity_factor=1.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.capacity_factor = capacity_factor
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # 负载均衡损失权重
        self.balance_loss_weight = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 计算每个样本分配给各个专家的概率
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # 计算Top-1或Top-2专家选择
        # 这里使用Top-1选择以简化示例
        top_k = 1
        top_gate_probs, top_experts = torch.topk(gate_probs, top_k, dim=1)
        
        # 计算容量限制（防止专家过载）
        capacity = int(self.capacity_factor * batch_size / self.num_experts)
        
        # 初始化输出和掩码
        output = torch.zeros_like(x[:, :self.output_dim])
        expert_masks = torch.zeros(self.num_experts, batch_size, device=x.device)
        
        # 为每个专家收集分配的样本
        for i in range(self.num_experts):
            # 找出分配给当前专家的样本
            expert_mask = (top_experts == i).any(dim=1)
            expert_samples = x[expert_mask]
            
            # 如果超过容量，随机选择部分样本
            if expert_samples.shape[0] > capacity:
                indices = torch.randperm(expert_samples.shape[0], device=x.device)[:capacity]
                expert_samples = expert_samples[indices]
                expert_mask[expert_mask.nonzero()[indices]] = True
            
            expert_masks[i] = expert_mask.float()
            
            # 专家处理样本
            if expert_samples.numel() > 0:
                expert_output = self.experts[i](expert_samples)
                output[expert_mask] = expert_output
        
        # 计算负载均衡损失
        # 基于每个专家的负载与平均负载的差异
        expert_loads = expert_masks.sum(dim=1)
        avg_load = expert_loads.mean()
        balance_loss = torch.sum((expert_loads - avg_load) **2) / self.num_experts
        
        return output, balance_loss * self.balance_loss_weight

# 训练循环中使用
def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    
    # 前向传播，获取模型输出和负载均衡损失
    output, balance_loss = model(x)
    
    # 计算主任务损失
    task_loss = F.cross_entropy(output, y)
    
    # 总损失 = 任务损失 + 负载均衡损失
    total_loss = task_loss + balance_loss
    
    # 反向传播和优化
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'balance_loss': balance_loss.item()
    }

# 使用示例
if __name__ == "__main__":
    # 超参数
    input_dim = 512
    output_dim = 10
    num_experts = 4
    expert_dim = 1024
    batch_size = 64
    
    # 初始化模型和优化器
    model = BalancedMoE(input_dim, output_dim, num_experts, expert_dim)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 模拟数据
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    
    # 训练步骤
    losses = train_step(model, x, y)
    print(f"损失: {losses}")


class GlobalBalancedMoE(BalancedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 增加滑动窗口记录历史负载
        self.load_history = torch.zeros(self.num_experts, 10)  # 记录10个批次的负载
        self.history_ptr = 0
    
    def forward(self, x):
        output, balance_loss = super().forward(x)
        
        # 记录当前负载
        expert_loads = self.load_history.sum(dim=1)
        self.load_history[:, self.history_ptr] = expert_loads
        self.history_ptr = (self.history_ptr + 1) % 10
        
        # 计算全局负载均衡损失（基于历史负载）
        avg_historical_load = self.load_history.mean(dim=1)
        global_balance_loss = torch.sum((avg_historical_load - avg_historical_load.mean())** 2)
        
        # 综合局部和全局损失
        total_balance_loss = balance_loss + 0.1 * global_balance_loss
        
        return output, total_balance_loss