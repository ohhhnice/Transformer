import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import spacy
import os
from tqdm import tqdm
import random
import math
from sklearn.model_selection import train_test_split

# 设置随机种子确保可复现性
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 加载分词器
def load_tokenizers():
    """加载英文和中文分词器"""
    try:
        spacy_en = spacy.load('en_core_web_sm')
    except:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load('en_core_web_sm')
    
    try:
        spacy_zh = spacy.load('zh_core_web_sm')
    except:
        os.system("python -m spacy download zh_core_web_sm")
        spacy_zh = spacy.load('zh_core_web_sm')
    
    return spacy_en, spacy_zh

# 分词函数
def tokenize_en(text, spacy_en):
    """英文分词，返回小写词列表"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text, spacy_zh):
    """中文分词，返回词列表"""
    return [tok.text for tok in spacy_zh.tokenizer(text)]

# 词汇表类
class Vocabulary:
    def __init__(self, min_freq=1):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_freq = min_freq
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, tokenized_sentences):
        """从分词后的句子列表构建词汇表"""
        frequencies = {}
        idx = 4  # 从4开始，0-3为特殊符号
        
        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in frequencies:
                    frequencies[token] = 1
                else:
                    frequencies[token] += 1
                    
                # 当达到最小频率时加入词汇表
                if frequencies[token] == self.min_freq:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
    
    def numericalize(self, tokenized_sentence):
        """将分词后的句子转换为数字序列"""
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_sentence
        ]

# 翻译数据集类
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, 
                 src_tokenizer, trg_tokenizer, max_len=50):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, index):
        src_sentence = self.src_sentences[index]
        trg_sentence = self.trg_sentences[index]
        
        # 分词
        src_tokenized = self.src_tokenizer(src_sentence)
        trg_tokenized = self.trg_tokenizer(trg_sentence)
        
        # 添加开始和结束标记
        src_tokenized = ["<SOS>"] + src_tokenized + ["<EOS>"]
        trg_tokenized = ["<SOS>"] + trg_tokenized + ["<EOS>"]
        
        # 转换为数字序列
        src_numericalized = self.src_vocab.numericalize(src_tokenized)
        trg_numericalized = self.trg_vocab.numericalize(trg_tokenized)
        
        # 填充或截断到最大长度
        src_padded = self.pad_sequence(src_numericalized)
        trg_padded = self.pad_sequence(trg_numericalized)
        
        return torch.tensor(src_padded), torch.tensor(trg_padded)
    
    def pad_sequence(self, sequence):
        """填充序列到最大长度"""
        if len(sequence) < self.max_len:
            sequence += [self.src_vocab.stoi["<PAD>"]] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        return sequence

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分多头
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力到值
        x = torch.matmul(attn, V)
        
        # 拼接多头结果
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        x = self.w_o(x)
        return x, attn

# 前馈神经网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩码自注意力子层
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 编码器-解码器注意力子层
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask):
        # 词嵌入并缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 添加位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_ff, num_layers, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, enc_output, self_mask, cross_mask):
        # 词嵌入并缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        # 添加位置编码
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
            
        # 输出层
        output = self.fc_out(x)
        return output

# Transformer模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, src_mask, trg_mask, cross_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, cross_mask)
        return dec_output

# 创建掩码
def create_pad_mask(seq, pad_idx):
    """创建填充掩码，掩盖填充符号"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_trg_mask(trg, pad_idx):
    """创建目标序列掩码，既掩盖填充符号，又掩盖未来信息"""
    # 填充掩码
    trg_pad_mask = create_pad_mask(trg, pad_idx)
    
    # 后续词掩码（下三角矩阵）
    trg_len = trg.size(1)
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()
    
    # 合并掩码
    return trg_pad_mask & trg_sub_mask

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, src_pad_idx, trg_pad_idx, device):
    model.train()
    epoch_loss = 0
    
    for src, trg in tqdm(train_loader, desc="Training"):
        src = src.to(device)
        trg = trg.to(device)
        
        # 目标输入（除最后一个token）和目标输出（除第一个token）
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        # 创建掩码
        src_mask = create_pad_mask(src, src_pad_idx)
        trg_mask = create_trg_mask(trg_input, trg_pad_idx)
        cross_mask = src_mask
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, trg_input, src_mask, trg_mask, cross_mask)
        
        # 计算损失
        output_dim = output.size(-1)
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)
        
        loss = criterion(output, trg_output)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)

# 评估函数
def evaluate(model, val_loader, criterion, src_pad_idx, trg_pad_idx, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in tqdm(val_loader, desc="Evaluating"):
            src = src.to(device)
            trg = trg.to(device)
            
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]
            
            src_mask = create_pad_mask(src, src_pad_idx)
            trg_mask = create_trg_mask(trg_input, trg_pad_idx)
            cross_mask = src_mask
            
            output = model(src, trg_input, src_mask, trg_mask, cross_mask)
            
            output_dim = output.size(-1)
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)
            
            loss = criterion(output, trg_output)
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)

# 翻译函数
def translate_sentence(model, sentence, src_vocab, trg_vocab, src_tokenizer, 
                       device, max_len=50):
    model.eval()
    
    # 分词处理
    if isinstance(sentence, str):
        tokens = src_tokenizer(sentence)
    else:
        tokens = sentence
    
    # 添加特殊标记
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    
    # 转换为数字序列
    numericalized = src_vocab.numericalize(tokens)
    
    # 填充序列
    padded = [0] * max_len
    padded[:len(numericalized)] = numericalized[:max_len]
    src_tensor = torch.LongTensor(padded).unsqueeze(0).to(device)
    
    # 创建源掩码
    src_mask = create_pad_mask(src_tensor, src_vocab.stoi["<PAD>"])
    
    # 编码器输出
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
    
    # 初始化目标序列
    trg_tokens = ["<SOS>"]
    trg_numericalized = [trg_vocab.stoi["<SOS>"]]
    
    # 解码过程
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_numericalized).unsqueeze(0).to(device)
        trg_mask = create_trg_mask(trg_tensor, trg_vocab.stoi["<PAD>"])
        cross_mask = src_mask
        
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_output, trg_mask, cross_mask)
        
        # 获取最后一个token的预测
        pred_token_idx = output[:, -1, :].argmax(1).item()
        trg_numericalized.append(pred_token_idx)
        trg_tokens.append(trg_vocab.itos[pred_token_idx])
        
        # 如果预测到结束标记则停止
        if pred_token_idx == trg_vocab.stoi["<EOS>"]:
            break
    
    return trg_tokens[1:-1]  # 去除<SOS>和<EOS>

# 主函数
def main():
    # 配置参数
    config = {
        'batch_size': 32,
        'd_model': 256,
        'nhead': 8,
        'd_ff': 512,
        'num_layers': 3,
        'dropout': 0.1,
        'lr': 0.0005,
        'num_epochs': 20,
        'max_len': 50,
        'min_freq': 1
    }
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载分词器
    spacy_en, spacy_zh = load_tokenizers()
    
    # 定义分词函数（绑定分词器）
    def en_tokenizer(text):
        return tokenize_en(text, spacy_en)
    
    def zh_tokenizer(text):
        return tokenize_zh(text, spacy_zh)
    
    # 示例数据（实际应用中应替换为大规模平行语料库）
    # 英文-中文平行语料
    english_sentences = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is John.",
        "I like to study Chinese.",
        "This is a transformer model.",
        "Machine translation is interesting.",
        "I want to learn more about AI.",
        "What time is it?",
        "It is three o'clock in the afternoon.",
        "I love eating Chinese food.",
        "Beijing is the capital of China.",
        "She is reading a book.",
        "We will go to the park tomorrow.",
        "He speaks English and Chinese."
    ]
    
    chinese_sentences = [
        "你好，你怎么样？",
        "我很好，谢谢你。",
        "你叫什么名字？",
        "我的名字是约翰。",
        "我喜欢学习中文。",
        "这是一个Transformer模型。",
        "机器翻译很有趣。",
        "我想学习更多关于人工智能的知识。",
        "现在几点了？",
        "现在是下午三点钟。",
        "我喜欢吃中国菜。",
        "北京是中国的首都。",
        "她正在读一本书。",
        "我们明天要去公园。",
        "他会说英语和中文。"
    ]
    
    # 构建词汇表
    en_vocab = Vocabulary(min_freq=config['min_freq'])
    zh_vocab = Vocabulary(min_freq=config['min_freq'])
    
    # 分词所有句子来构建词汇表
    en_tokenized = [en_tokenizer(sent) for sent in english_sentences]
    zh_tokenized = [zh_tokenizer(sent) for sent in chinese_sentences]
    
    en_vocab.build_vocabulary(en_tokenized)
    zh_vocab.build_vocabulary(zh_tokenized)
    
    print(f"英文词汇表大小: {len(en_vocab)}")
    print(f"中文词汇表大小: {len(zh_vocab)}")
    
    # 划分训练集和验证集
    train_en, val_en, train_zh, val_zh = train_test_split(
        english_sentences, chinese_sentences, test_size=0.2, random_state=SEED
    )
    
    # 创建数据集和数据加载器
    train_dataset = TranslationDataset(
        train_en, train_zh, en_vocab, zh_vocab,
        en_tokenizer, zh_tokenizer, config['max_len']
    )
    
    val_dataset = TranslationDataset(
        val_en, val_zh, en_vocab, zh_vocab,
        en_tokenizer, zh_tokenizer, config['max_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # 初始化模型
    encoder = Encoder(
        vocab_size=len(en_vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    decoder = Decoder(
        vocab_size=len(zh_vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    model = Transformer(encoder, decoder).to(device)
    
    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    
    # 训练模型
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            en_vocab.stoi["<PAD>"], zh_vocab.stoi["<PAD>"],
            device
        )
        
        val_loss = evaluate(
            model, val_loader, criterion,
            en_vocab.stoi["<PAD>"], zh_vocab.stoi["<PAD>"],
            device
        )
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')
            print("保存最佳模型")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_transformer.pth'))
    
    # 测试翻译
    test_sentences = [
        "Hello, world!",
        "I love China.",
        "This is a test.",
        "What is your favorite food?",
        "I study computer science."
    ]
    
    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, en_vocab, zh_vocab, en_tokenizer, device
        )
        print(f"英文: {sentence}")
        print(f"中文: {' '.join(translation)}")
        print("---")

if __name__ == "__main__":
    main()
