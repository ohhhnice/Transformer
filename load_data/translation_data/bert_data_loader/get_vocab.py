import json
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
            if count >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def save(self, filepath_word2idx, filepath_idx2word):
        with open(filepath_word2idx, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=4)
        with open(filepath_idx2word, 'w', encoding='utf-8') as f:
            json.dump(self.idx2word, f, ensure_ascii=False, indent=4)