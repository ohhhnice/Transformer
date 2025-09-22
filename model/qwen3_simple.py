from transformers import AutoModelForCausalLM, AutoTokenizer

# 选择模型版本，例如 4B 指令微调版本
model_name = "Qwen/Qwen3-4B-Instruct-2507"  # 也可选择 "Qwen/Qwen3-8B", "Qwen/Qwen3-32B" 等

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # 自动分配 GPU 和 CPU
    torch_dtype="auto",  # 自动选择数据类型
)

# 构建输入消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用Python写一个快速排序算法。"}
]

# 应用聊天模板并 tokenize
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成输出
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.8
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)