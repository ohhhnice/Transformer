from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_pth_path = "./model/qwen3_4b/model_pth/"
tokenizer = AutoTokenizer.from_pretrained(model_pth_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_pth_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# 准备输入
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 生成文本并捕获中间输出（关键参数配置）
generated_output = model.generate(
    **model_inputs,
    max_new_tokens=32768,  # 建议先减小值测试，32768可能导致内存溢出
    return_dict_in_generate=True,  # 关键：返回字典格式的完整输出
    output_scores=True,  # 关键：返回每个步骤的logits
    output_attentions=True,  # 如需注意力权重可设为True（增加内存占用）
    output_hidden_states=True  # 如需隐藏状态可设为True（大幅增加内存占用）
)

# --------------------------
# 提取中间输出（用于蒸馏）
# --------------------------

# 1. 最终生成的文本
generated_ids = generated_output.sequences
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("最终生成文本：\n", generated_text)

# 2. 中间步骤的logits（核心蒸馏信号）
# scores是一个元组，每个元素是该步骤的logits，形状为 (batch_size, vocab_size)
intermediate_scores = generated_output.scores  # 长度 = max_new_tokens（实际生成的token数）

# 示例：打印前3步的logits形状
for i, logits in enumerate(intermediate_scores[:3]):
    print(f"第{i+1}步logits形状：", logits.shape)  # 输出：torch.Size([1, vocab_size])

# 3. 将logits转换为概率分布（可选，蒸馏时常用）
intermediate_probs = [torch.softmax(logits, dim=-1) for logits in intermediate_scores]

# --------------------------
# 蒸馏场景下的使用示例
# --------------------------
"""
在蒸馏中，教师模型（当前模型）的中间logits/probs可用于指导学生模型学习：
1. 学生模型生成同样的序列，输出自己的logits
2. 计算教师logits与学生logits的损失（如KL散度）
3. 结合学生的生成损失（交叉熵），共同优化学生模型
"""
# 伪代码示意：
# student_probs = student_model.generate(..., output_scores=True).scores
# distillation_loss = torch.nn.KLDivLoss()(student_probs.log(), intermediate_probs)
# total_loss = generation_loss + alpha * distillation_loss
    