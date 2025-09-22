from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器（需要确保你有访问权限）
model_name = "Qwen/Qwen3-4B"  # 根据你的需求选择不同大小的模型

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 生成文本
prompt = "人工智能是"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"生成结果: {result}")