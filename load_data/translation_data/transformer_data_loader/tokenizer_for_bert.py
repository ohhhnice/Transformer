from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import torch



def translation_dataloader_huge(batch_size, src_max_len, tgt_max_len,
                                train_json_file, val_json_file,
                                encoder_vocab, decoder_vocab, tokenizer):
    # 直接返回数据加载器
    train_data = TranslationDatasetHuge(data_file=train_json_file, encoder_vocab=encoder_vocab, 
                                        decoder_vocab=decoder_vocab, tokenizer=tokenizer, 
                                        src_max_len=src_max_len, tgt_max_len=tgt_max_len)
    test_data = TranslationDatasetHuge(data_file=val_json_file, encoder_vocab=encoder_vocab,
                                       decoder_vocab=decoder_vocab, tokenizer=tokenizer, 
                                        src_max_len=src_max_len, tgt_max_len=tgt_max_len)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

class TranslationDatasetHuge(Dataset):
    def __init__(self, data_file, encoder_vocab, decoder_vocab, tokenizer, src_max_len, tgt_max_len):
        self.data_file = data_file
        self.encoder_vocab = encoder_vocab
        self.decoder_vocab = decoder_vocab
        self.tokenizer = tokenizer
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
        zh_tokenized = self.tokenizer(zh_sentence)
        en_tokenized = self.tokenizer(en_sentence)
        zh_sentence = zh_tokenized
        en_sentence = en_tokenized
        zh_indices = self.decoder_vocab.numericalize(["[CLS]"] + zh_sentence[:self.src_max_len-2] + ["[SEP]"])
        en_indices = self.encoder_vocab.convert_tokens_to_ids(["[CLS]"] + en_sentence[:self.tgt_max_len-2] + ["[SEP]"])
        zh_indices_pad = zh_indices + [self.decoder_vocab.word2idx.get("[PAD]")] * (self.src_max_len - len(zh_indices))
        en_indices_pad = en_indices + [self.encoder_vocab.convert_tokens_to_ids("[PAD]")] * (self.tgt_max_len - len(en_indices))
        return torch.tensor(en_indices_pad), torch.tensor(zh_indices_pad)