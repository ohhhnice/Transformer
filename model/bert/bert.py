import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from typing import Optional, Tuple


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, pad_token_id=0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        
        # 注册position_ids为缓冲区（非可训练参数，无需从权重加载）
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).unsqueeze(0))

    def forward(self, input_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        
        input_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = input_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"隐藏层大小 {hidden_size} 必须能被注意力头数 {num_attention_heads} 整除")
            
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 自定义模型的Q/K/V层名称：attention.query（无self.）
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32, device=hidden_states.device))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer


class BertAttentionOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.attention_output = BertAttentionOutput(hidden_size)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output, hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size) 
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self, 
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 pad_token_id=0):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            pad_token_id=pad_token_id
        )
        self.encoder = BertEncoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=3072
        )
        self.pooler = BertPooler(hidden_size=hidden_size)
        
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, 
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask
        )
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output

    def load_safetensors_weights(self, safetensors_paths: list):
        """
        修复点：
        1. 处理attention.self.→attention.的名称映射
        2. 忽略缓冲区参数embeddings.position_ids的检查
        """
        # 1. 合并多文件权重
        full_state_dict = {}
        for path in safetensors_paths:
            partial_state_dict = load_file(path)
            full_state_dict.update(partial_state_dict)
        print(f"已加载 {len(safetensors_paths)} 个文件，共 {len(full_state_dict)} 个参数")

        # 2. 权重名称映射（核心修复：新增attention.self.→attention.的替换）
        new_state_dict = {}
        for old_key, value in full_state_dict.items():
            # 跳过CLS头（下游任务层，基础模型不需要）
            if old_key.startswith("cls."):
                continue
            
            # 映射1：移除bert.前缀
            new_key = old_key.replace("bert.", "")
            # 映射2：attention.self.→attention.（解决Q/K/V层名称不匹配）
            new_key = new_key.replace("attention.self.", "attention.")
            # 映射3：LayerNorm的gamma→weight、beta→bias
            new_key = new_key.replace(".gamma", ".weight")
            new_key = new_key.replace(".beta", ".bias")
            
            new_state_dict[new_key] = value

        # 3. 加载权重：指定strict=False，忽略缓冲区参数embeddings.position_ids的检查
        # （因为position_ids是register_buffer注册的，不在权重文件中）
        self.load_state_dict(new_state_dict, strict=False)
        
        # 验证关键参数是否加载成功（可选，用于调试）
        required_keys = [
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.query.weight",
            "encoder.layer.11.output.LayerNorm.weight"
        ]
        missing_required = [key for key in required_keys if key not in new_state_dict]
        if missing_required:
            print(f"警告：关键参数缺失 {missing_required}")
        else:
            print("权重加载成功！所有关键参数已匹配")


# 使用示例
if __name__ == "__main__":
    # 1. 配置BERT-base-uncased参数
    BERT_CONFIG = {
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "pad_token_id": 0
    }

    # 2. 创建模型
    model = BertModel(**BERT_CONFIG)

    # 3. 加载权重（替换为你的safetensors文件路径）
    # 注意：确保路径正确，且包含2个文件（model-00001-of-00002.safetensors和model-00002-of-00002.safetensors）
    safetensors_paths = [
        "./model/bert/model.safetensors",
    ]
    try:
        model.load_safetensors_weights(safetensors_paths)
    except Exception as e:
        print(f"加载失败: {e}")
        exit(1)

    # 4. 测试模型（输入必须为LongTensor）
    input_ids = torch.zeros((2, 128), dtype=torch.long)  # 批次2，序列长度128
    attention_mask = torch.ones(2, 128, dtype=torch.float32)        # 无padding，全1掩码
    token_type_ids = torch.zeros(2, 128, dtype=torch.long)          # 单段落，全0

    # 关闭梯度计算，加速测试
    with torch.no_grad():
        sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)

    # 验证输出形状（符合预期则说明模型正常）
    assert sequence_output.shape == (2, 128, 768), f"序列输出形状错误，预期(2,128,768)，实际{sequence_output.shape}"
    assert pooled_output.shape == (2, 768), f"池化输出形状错误，预期(2,768)，实际{pooled_output.shape}"
    print(f"序列输出形状: {sequence_output.shape}")  # (2, 128, 768)
    print(f"池化输出形状: {pooled_output.shape}")    # (2, 768)
    print("模型测试成功！")
    print(sequence_output[0][0][:2])