from torch.utils.data import Dataset
import torch

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
     
    return english_sentences, chinese_sentences


class Vocabulary():
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, tokenized_sentences):
        freq = {}
        for sentence in tokenized_sentences:
            # 统计频次
            for token in sentence:
                if token not in freq:
                    freq[token] = 1
                else:
                    freq[token] += 1
            
            # 入词汇表
        for token, count in freq.items():
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
    
    def numericalize(self, tokenized_sentence):
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokenized_sentence]
        
        

class TranslationDataset(Dataset):
    def __init__(self, data, en_vocab, zh_vocab, max_len=30):
        self.data = data
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en_sentence, zh_sentence = self.data[idx]
        en_sentence = ["<SOS>"] + en_sentence + ["<EOS>"]
        zh_sentence = ["<SOS>"] + zh_sentence + ["<EOS>"]
        en_indices = self.en_vocab.numericalize(en_sentence)
        zh_indices = self.zh_vocab.numericalize(zh_sentence)
        en_indices_pad = en_indices[:self.max_len] + [self.en_vocab.word2idx["<PAD>"]] * (self.max_len - len(en_indices))
        zh_indices_pad = zh_indices[:self.max_len] + [self.zh_vocab.word2idx["<PAD>"]] * (self.max_len - len(zh_indices))
        return torch.tensor(en_indices_pad), torch.tensor(zh_indices_pad)