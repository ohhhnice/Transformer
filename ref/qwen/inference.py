import torch
from model import Qwen3ForCausalLM
from config import Qwen3Config
from tokenizer import Qwen3Tokenizer

def generate_text(prompt, max_length=100):
    # 加载配置和模型
    config = Qwen3Config()
    tokenizer = Qwen3Tokenizer()
    model = Qwen3ForCausalLM(config)
    
    # 加载预训练权重（如果有）
    # model.load_state_dict(torch.load("path/to/pretrained/model.pt"))
    
    model.eval()
    
    # 编码输入
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 使用温度采样
            temperature = 0.8
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / temperature, dim=-1), 1
            )
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 如果生成了结束标记，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

if __name__ == "__main__":
    prompt = "人工智能是"
    result = generate_text(prompt)
    print(f"输入: {prompt}")
    print(f"生成: {result}")