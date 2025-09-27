from torch import nn
from .decoder_block import DecoderBlock
import torch
from utils.moe_loss import load_balancing_loss_func

class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, mask=None):
        '''
        x (b, s)
        mask (b, s, s)
        '''
        all_router_logits = []
        x = self.word_embedding(x)
        for layer in self.decoder:
            x, router_logits = layer(x, mask)
            if router_logits is not None:  # 只收集MoE层的logits
                all_router_logits.append(router_logits)
        x = self.fc_out(x)

        # 计算负载均衡损失（仅当存在MoE层时）
        balance_loss = torch.tensor(0.0, device=x.device)
        if all_router_logits and mask is not None:
            two_dim_mask = mask.diagonal(dim1=2, dim2=3).squeeze()
            # 调用负载均衡损失函数
            balance_loss = load_balancing_loss_func(
                gate_logits=tuple(all_router_logits),  # 转换为元组，符合函数输入要求
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                attention_mask=two_dim_mask  # 传入有效token掩码
            )
            # 应用损失权重
            balance_loss = balance_loss
        return x, balance_loss
        