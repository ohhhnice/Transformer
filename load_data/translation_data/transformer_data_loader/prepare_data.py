from torch.utils.data import Dataset
import torch
import json

def get_data_big(json_file):
    english_sentences, chinese_sentences = [], []
    with open(json_file, "r", encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            english_sentences.append(item['english'])
            chinese_sentences.append(item['chinese'])
    return chinese_sentences, english_sentences

def get_data():
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
     
    return chinese_sentences, english_sentences


class Vocabulary():
    def __init__(self, min_freq=50):
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.min_freq = min_freq
        self.freq = {}

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, tokenized_sentences):
        # 统计频次
        for sentence in tokenized_sentences:
            self.add_sentence(sentence)
        self.build()    # 入词汇表

    def load_from_file(self, filepath_word2idx, filepath_idx2word):
        with open(filepath_word2idx, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        with open(filepath_idx2word, 'r', encoding='utf-8') as f:
            idx2word_str_keys = json.load(f)
            self.idx2word = {int(idx): word for idx, word in idx2word_str_keys.items()}
    
    def add_sentence(self, tokenized_sentence):
        for token in tokenized_sentence:
            if token not in self.freq:
                self.freq[token] = 1
            else:
                self.freq[token] += 1
    
    def build(self):
        for token, count in self.freq.items():
            if count >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
    
    def numericalize(self, tokenized_sentence):
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokenized_sentence]
    
    def save(self, 
            filepath_idx2word="./load_data/translation_data/vocab/idx2word.json", 
            filepath_word2idx="./load_data/translation_data/vocab/word2idx.json"):
        with open(filepath_word2idx, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
        with open(filepath_idx2word.replace('word2idx', 'idx2word'), 'w', encoding='utf-8') as f:
            json.dump(self.idx2word, f, ensure_ascii=False, indent=4)
        
        

class TranslationDataset(Dataset):
    def __init__(self, data, zh_vocab, en_vocab, max_len):
        self.data = data
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        zh_sentence, en_sentence = self.data[idx]
        zh_sentence = ["<SOS>"] + zh_sentence + ["<EOS>"]
        en_sentence = ["<SOS>"] + en_sentence + ["<EOS>"]
        zh_indices = self.zh_vocab.numericalize(zh_sentence)
        en_indices = self.en_vocab.numericalize(en_sentence)
        zh_indices_pad = zh_indices[:self.max_len] + [self.zh_vocab.word2idx["<PAD>"]] * (self.max_len - len(zh_indices))
        en_indices_pad = en_indices[:self.max_len] + [self.en_vocab.word2idx["<PAD>"]] * (self.max_len - len(en_indices))
        return torch.tensor(zh_indices_pad), torch.tensor(en_indices_pad)
    
class TranslationDatasetHuge(Dataset):
    def __init__(self, data_file, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer, src_max_len, tgt_max_len):
        self.data_file = data_file
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.zh_tokenizer = zh_tokenizer
        self.en_tokenizer = en_tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.num_samples = self._count_lines()
        self.offsets = self._build_offsets()
        
    def _count_lines(self):
        """计算文件总行数（即样本数）"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
        
    def _build_offsets(self):
        """记录每行的起始字节位置，加速按索引读取"""
        offsets = [0]  # 第一行从0开始
        with open(self.data_file, 'r', encoding='utf-8') as f:
            while f.readline():
                offsets.append(f.tell())  # 记录当前位置（下一行的起始）
        return offsets

    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """按索引读取单行JSON并解析"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])  # 直接跳转到目标行
            line = f.readline()        # 读取一行
        
        data = json.loads(line)
        zh_sentence, en_sentence = data["chinese"], data["english"]
        zh_tokenized = self.zh_tokenizer(zh_sentence)
        en_tokenized = self.en_tokenizer(en_sentence)
        zh_sentence = ["<SOS>"] + zh_tokenized + ["<EOS>"]
        en_sentence = ["<SOS>"] + en_tokenized + ["<EOS>"]
        zh_indices = self.zh_vocab.numericalize(zh_sentence)
        en_indices = self.en_vocab.numericalize(en_sentence)
        zh_indices_pad = zh_indices[:self.src_max_len] + [self.zh_vocab.word2idx["<PAD>"]] * (self.src_max_len - len(zh_indices))
        en_indices_pad = en_indices[:self.tgt_max_len] + [self.en_vocab.word2idx["<PAD>"]] * (self.tgt_max_len - len(en_indices))
        return torch.tensor(zh_indices_pad), torch.tensor(en_indices_pad)