import torch
from torch.utils.data import Dataset, DataLoader
from model import Qwen3ForCausalLM
from config import Qwen3Config
from tokenizer import Qwen3Tokenizer
import os

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        self.tokenized_text = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokenized_text) // self.block_size
        
    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        tokens = self.tokenized_text[start_idx:end_idx]
        return torch.tensor(tokens, dtype=torch.long)

def train():
    # 配置参数
    config = Qwen3Config()
    tokenizer = Qwen3Tokenizer()
    
    # 初始化模型
    model = Qwen3ForCausalLM(config)
    
    # 数据加载
    dataset = TextDataset("data/train.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
                
        print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataloader)}")
        
        # 保存检查点
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()