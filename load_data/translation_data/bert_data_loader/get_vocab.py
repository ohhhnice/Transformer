import json
from tqdm import tqdm

class Vocab():
    def __init__(self, min_freq=5):
        self.idx2word = {0:"[PAD]", 1:"[CLS]", 2:"[SEP]", 3:"[UNK]", 4:"[MASK]"} 
        self.word2idx = {"[PAD]":0, "[CLS]":1, "[SEP]":2, "[UNK]":3, "[MASK]":4}
        self.freq = {}
        self.min_freq = min_freq

    def add_sentence(self, tokenized_sentences):
        for token in tokenized_sentences:
            if token not in self.freq:
                self.freq[token] = 1
            else:
                self.freq[token] += 1

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self):
        for token, count in self.freq.items():
            if count >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def numericalize(self, tokens):
        return [self.word2idx.get(token, self.word2idx["[UNK]"]) for token in tokens]
    
    def concate_sentences(self, tokens_a, tokens_b, max_len):
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:(max_len - len(tokens_b) - 3)]
            else:
                tokens_b = tokens_b[:(max_len - len(tokens_a) - 3)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        tokens = tokens + ["[PAD]"] * (max_len - len(tokens))
        tokens_ids = self.numericalize(tokens)
        seg_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        attn_mask = [1] * len(seg_ids) + [0] * (max_len - len(seg_ids))
        return tokens, tokens_ids, seg_ids, attn_mask

    def load_from_file(self, filepath_idx2word, filepath_word2idx):
        with open(filepath_word2idx, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
        with open(filepath_idx2word, 'r', encoding='utf-8') as f:
            idx2word_str_keys = json.load(f)
            self.idx2word = {int(idx): word for idx, word in idx2word_str_keys.items()}

    def save(self, filepath_word2idx, filepath_idx2word):
        with open(filepath_word2idx, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
        with open(filepath_idx2word, 'w', encoding='utf-8') as f:
            json.dump(self.idx2word, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    import sys
    import os
    from pathlib import Path
    current_dir = os.path.abspath(__file__)
    root_dir = str(Path(current_dir).parents[3])
    sys.path.insert(0, root_dir)
    from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
    tokenizer = get_multilingual_tokenizer()
    train_file = "./load_data/translation_data/translation2019zh/translation2019zh_train.json"
    val_file = "./load_data/translation_data/translation2019zh/translation2019zh_valid.json"
    vocab = Vocab(min_freq=50)
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building Train Vocab", total=5161434):
            json_line = json.loads(line)
            zh_sentence, en_sentence = json_line["chinese"], json_line["english"]
            zh_sentence_tokenized = tokenizer.tokenize(zh_sentence)  
            en_sentence_tokenized = tokenizer.tokenize(en_sentence)
            vocab.add_sentence(zh_sentence_tokenized)
            vocab.add_sentence(en_sentence_tokenized)
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building Val Vocab", total = 39323):
            json_line = json.loads(line)
            zh_sentence, en_sentence = json_line["chinese"], json_line["english"]
            zh_sentence_tokenized = tokenizer.tokenize(zh_sentence)
            en_sentence_tokenized = tokenizer.tokenize(en_sentence)
            vocab.add_sentence(zh_sentence_tokenized)
            vocab.add_sentence(en_sentence_tokenized)
    vocab.build_vocab()
    print(f"Vocab size: {len(vocab)}")
    vocab.save(
        filepath_word2idx="./load_data/translation_data/bert_data_loader/vocab/word2idx.json", 
        filepath_idx2word="./load_data/translation_data/bert_data_loader/vocab/idx2word.json"
    )