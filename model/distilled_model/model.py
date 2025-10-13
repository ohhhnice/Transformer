from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

# 如果你有本地的config文件

config_path = "./model/qwen3-0_6B/"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_config = AutoConfig.from_pretrained(config_path)
model = AutoModelForCausalLM.from_config(model_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(config_path)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

model.apply(init_weights)


config = {"output_hidden_states":True, "output_attentions":True}

with torch.no_grad():outputs = model(model_inputs["input_ids"], **config)

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)