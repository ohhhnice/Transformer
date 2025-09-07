from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT模型和分词器
model_name = "bert-base-uncased"  # 基础的英文无大小写模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 输入文本
text = "Hello, this is a BERT model example."

# 文本预处理：分词、添加特殊符号、转换为ID、生成注意力掩码
inputs = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"  # 返回PyTorch张量
)

# 模型推理（不计算梯度以提高效率）
with torch.no_grad():
    outputs = model(**inputs)

# 获取最后一层的隐藏状态 (batch_size, sequence_length, hidden_size)
last_hidden_states = outputs.last_hidden_state

# 获取[CLS] token的嵌入作为句子表示 (通常用于句子级任务)
cls_embedding = last_hidden_states[:, 0, :]

print("输入文本的分词结果:", tokenizer.tokenize(text))
print("句子嵌入的形状:", cls_embedding.shape)  # 应为 (1, 768) 对于bert-base模型