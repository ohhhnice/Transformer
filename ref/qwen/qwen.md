blog:https://blog.csdn.net/qq_37021523/article/details/138901191
知乎:https://zhuanlan.zhihu.com/p/1901014191235633835


虽然我目前无法提供完整且可直接生产环境使用的Qwen3 MoE模型代码（因为这通常涉及大量代码和详细配置），但我可以为你介绍其核心组件和关键代码片段，并说明如何搭建其主要结构。Qwen3的MoE模型采用了典型的**Decoder-Only Transformer架构**，其中一些层的MLP被替换为**稀疏MoE块（Sparse MoE Block）**。

# 🧠 Qwen3 MoE模型核心组件与实现思路

以下是基于Hugging Face Transformers库风格的Qwen3 MoE关键部分实现要点：

## 🏗️ 1. 稀疏MoE块（Sparse MoE Block）

这是Qwen3 MoE的核心，它用多个专家（MLP）和一个门控网络（Gating Network）替代了传统的密集MLP层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts # 专家数量，例如8个
        self.top_k = config.num_experts_per_tok # 每个token使用的专家数，通常为2
        self.norm_topk_prob = config.norm_topk_prob # 是否对top-k权重进行归一化

        # 门控网络：线性层，输出维度为专家数量
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        # 专家列表：每个专家是一个MLP
        self.experts = nn.ModuleList([
            Qwen3MoeMLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim) # 重塑为 (batch_size * seq_len, hidden_dim)

        # 计算路由逻辑值
        router_logits = self.gate(hidden_states) # (batch_size * seq_len, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # 计算softmax得到路由权重

        # 选取top-k专家及其权重
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # 对top-k权重进行归一化

        routing_weights = routing_weights.to(hidden_states.dtype) # 转换回原始数据类型
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )

        # 生成专家掩码：用于标识哪些token被分配给了哪些专家
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 遍历所有专家，只计算被分配到的token
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx]) # 找到属于当前专家的token索引

            if top_x.shape[0] == 0:
                continue # 没有token被分配给此专家则跳过

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim) # 获取属于当前专家的token隐藏状态
            current_hidden_states = expert_layer(current_state) # 专家前向传播

            # 用路由权重加权专家输出，并通过index_add_累加到最终输出中
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits # 返回输出和路由logits（可用于辅助损失计算）
```
*代码参考自*

## 🧩 2. 专家MLP（Qwen3MoeMLP）

每个专家本质上是一个前馈神经网络，Qwen3通常使用SwiGLU激活函数。

```python
class Qwen3MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size # 专家中间层维度，通常比密集MLP大

        # SwiGLU激活函数的门控线性层
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU() # SiLU激活函数用于SwiGLU

    def forward(self, x):
        # SwiGLU: swish(x * W_gate) * (x * W_up)
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up) # 下投影回隐藏层维度
```

## ⚙️ 3. 解码器层（Decoder Layer）

Qwen3的解码器层包含自注意力机制和MoE前馈网络（或密集MLP）。

```python
class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(config) # 自注意力模块
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps) # RMSNorm归一化
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # 判断该层是否使用MoE
        if getattr(config, 'is_moe_layer', False):
            self.mlp = Qwen3MoeSparseMoeBlock(config) # 使用MoE块
        else:
            self.mlp = Qwen3MoeMLP(config) # 使用密集MLP

    def forward(self, hidden_states, attention_mask=None, position_ids=None, output_router_logits=False):
        # 自注意力子层
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states # 残差连接

        # MoE/MLP子层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if getattr(self, 'is_moe_layer', False):
            hidden_states, router_logits = self.mlp(hidden_states) # MoE层返回输出和路由logits
        else:
            hidden_states = self.mlp(hidden_states) # 密集MLP层只返回输出
        hidden_states = residual + hidden_states # 残差连接

        if output_router_logits and getattr(self, 'is_moe_layer', False):
            return hidden_states, router_logits
        return hidden_states
```
*注意力机制（Qwen3MoeAttention）的实现与标准Transformer类似，但可能包含QK归一化等细节，此处省略详细代码。*

## 🌐 4. 模型主体（Qwen3MoeModel）

模型主体负责组合嵌入层、多层解码器和最终归一化层。

```python
class Qwen3MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 旋转位置编码（RoPE）
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        
        # 应用RoPE位置编码（通常在注意力模块内部完成）
        # ... 注意力计算会调用self.rotary_emb ...

        # 逐层通过解码器
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states
```

## 💡 5. 语言建模头（For Causal LM）

用于因果语言建模任务的输出头。

```python
class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 负载均衡辅助损失系数
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(self, input_ids, labels=None, attention_mask=None, output_router_logits=False):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0] # 获取最后一层隐藏状态
        logits = self.lm_head(hidden_states) # 通过语言建模头得到词汇表上的logits

        loss = None
        aux_loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

            # 如果输出路由logits，计算负载均衡辅助损失
            if output_router_logits:
                # ... 收集所有层的router_logits并计算aux_loss ...
                # aux_loss = ... (计算专家负载均衡的损失)
                loss = loss + self.router_aux_loss_coef * aux_loss

        return {'loss': loss, 'logits': logits, 'aux_loss': aux_loss}
```

## 📊 Qwen3 MoE模型配置示例

```python
from dataclasses import dataclass

@dataclass
class Qwen3MoeConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_hidden_layers: int = 32 # 总层数
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    head_dim: int = hidden_size // num_attention_heads
    rms_norm_eps: float = 1e-6
    moe_intermediate_size: int = 11008 # 专家中间层大小
    intermediate_size: int = 11008 # 密集MLP中间层大小
    num_experts: int = 8 # 专家总数
    num_experts_per_tok: int = 2 # 每个token使用的专家数
    norm_topk_prob: bool = True # 是否对top-k权重归一化
    router_aux_loss_coef: float = 0.01 # 负载均衡辅助损失系数
    # ... 其他配置参数 ...
```

## 🚀 使用示例（前向传播）

```python
config = Qwen3MoeConfig()
model = Qwen3MoeForCausalLM(config)

# 假设我们有一些输入
input_ids = torch.randint(0, config.vocab_size, (2, 128)) # (batch_size, seq_len)
attention_mask = torch.ones_like(input_ids)

# 前向传播
outputs = model(input_ids, attention_mask=attention_mask, output_router_logits=True)
logits = outputs['logits']
loss = outputs['loss']
aux_loss = outputs['aux_loss']

print(f"Logits shape: {logits.shape}") # 应为 (2, 128, vocab_size)
if aux_loss is not None:
    print(f"Auxiliary loss: {aux_loss.item()}")
```

---

**请注意以下几点：**

1.  **完整性**：以上代码是**高度简化的示意图**，主要用于展示Qwen3 MoE的核心思想。完整的实现需要处理缓存（Cache）、精确的掩码构造、RoPE位置编码、梯度检查点等大量细节。
2.  **配置灵活性**：实际的Qwen3系列包含不同规模的模型（如0.6B, 1.7B, 4B, 8B, 32B Dense, 30B/235B MoE），其配置（层数、头数、专家数等）差异很大。
3.  **负载均衡损失**：MoE训练中的关键点是**辅助损失（Auxiliary Loss）**，它鼓励专家被均衡使用，防止某些专家被忽略而 others 过载。其具体计算方式需参考原始论文或代码。
4.  **框架依赖**：完整实现通常基于深度学习框架（如PyTorch, MindSpore）及其Transformer库。阿里云和MindSpore团队对Qwen3提供了支持。
5.  **官方资源**：对于生产或研究用途，强烈建议参考官方实现：
    *   **Hugging Face Transformers**：未来版本可能会正式集成Qwen3 MoE。
    *   **MindSpore ModelZoo**：[https://www.mindspore.cn/news/detail?id=3709](https://www.mindspore.cn/news/detail?id=3709) (MindSpore版Qwen3)
    *   **官方GitHub**：[https://github.com/QwenLM](https://github.com/QwenLM) (查找Qwen3代码库)
    *   **ModelScope / Hugging Face Hub**：获取预训练模型权重。

希望这些核心代码片段和说明能帮助你理解Qwen3 MoE模型的架构和实现思路！如果你想深入了解某个特定部分，或者有具体的应用场景，可以告诉我。