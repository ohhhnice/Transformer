import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

# 第一层嵌套：嵌入子模块（被主模型调用）
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入
        self.position_embedding = nn.Embedding(100, embed_dim)      # 位置嵌入（假设最大序列长度100）
        self.norm = nn.LayerNorm(embed_dim)  # 归一化层

# 第二层嵌套：注意力子模块（被编码器调用）
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # QKV合并投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)      # 输出投影
        self.num_heads = num_heads

# 第三层嵌套：编码器子模块（被主模型调用）
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)  # 嵌套注意力模块
        self.ffn = nn.Sequential(  # 前馈网络（也是一种嵌套形式）
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

# 主模型（包含所有嵌套模块）
class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, num_heads=4, num_classes=2):
        super().__init__()
        self.embedding = TextEmbedding(vocab_size, embed_dim)  # 嵌套嵌入模块
        self.encoder = TransformerEncoder(embed_dim, num_heads)  # 嵌套编码器模块
        self.classifier = nn.Linear(embed_dim, num_classes)  # 分类头

    def forward(self, input_ids):
        # 嵌入层
        batch_size, seq_len = input_ids.shape
        token_embeds = self.embedding.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.embedding.position_embedding(pos_ids)
        x = self.embedding.norm(token_embeds + pos_embeds)
        
        # 编码器
        x = self.encoder.attention(x)  # 调用嵌套的注意力模块
        x = self.encoder.ffn(x)        # 调用嵌套的前馈网络
        
        # 分类
        x = x.mean(dim=1)  # 序列平均池化
        return self.classifier(x)


# 演示：初始化模型并保存为safetensors
if __name__ == "__main__":
    # 1. 创建带嵌套结构的模型
    model = TextClassifier(
        vocab_size=1000,    # 词汇表大小
        embed_dim=128,      # 嵌入维度
        num_heads=4,        # 注意力头数
        num_classes=2       # 分类类别数
    )

    # 2. 获取模型参数（state_dict中的key就是safetensors的命名）
    state_dict = model.state_dict()
    print("嵌套模型的参数命名（safetensors将使用这些名称）：")
    for name, param in state_dict.items():
        print(f"  {name} → 形状: {param.shape}")

    # 3. 保存为safetensors文件
    save_file(state_dict, "nested_model.safetensors")
    print("\n模型已保存为 nested_model.safetensors")

    # 4. 验证加载（可选）
    loaded_state = load_file("nested_model.safetensors")
    new_model = TextClassifier()
    new_model.load_state_dict(loaded_state)  # 加载成功说明命名匹配
    print("safetensors加载成功，参数命名完全匹配")


