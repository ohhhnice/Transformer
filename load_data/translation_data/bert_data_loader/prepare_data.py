from torch.utils.data import Dataset
import json
class TranslationDataset(Dataset):
    def __init__(self, file_path, max_len, tokenizer, vocab, padding_idx):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab = vocab
        self.padding_idx = padding_idx

        self.num_samples = self._count_lines()
        self.offsets = self.build_index()

    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            json_line = json.loads(line)
            zh_sentence, en_sentence = json_line["chinese"], json_line["english"]
            zh_sentence_tokenized = self.tokenizer(zh_sentence)
            en_sentence_tokenized = self.tokenizer(en_sentence)
            merged_tokens, merged_tokens_ids, seg_ids, attn_mask = self.vocab.concate_sentences(zh_sentence_tokenized, en_sentence_tokenized, self.max_len)
            '''
            还有mask逻辑，还有label，len也要改（2倍数），一半是正例，一半是负例
            '''
        return merged_tokens, merged_tokens_ids, seg_ids, attn_mask

    def __len__(self):
        return len(self.num_samples)


    def _count_lines(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            return sum([1 for _ in f])*2

    def build_index(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            offsets = [0]
            while f.readline():
                offsets.append(f.tell())
        return offsets
    
    def _get_ids(self):
        pass

    def _get_attn_mask(self):
        pass
