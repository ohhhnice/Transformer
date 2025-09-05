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
            merged_tokens, merged_tokens_ids, seg_ids, attn_mask = self.concate_sentences(zh_sentence_tokenized, en_sentence_tokenized, self.max_len)
            '''
            还有mask逻辑，还有label，len也要改（2倍数），一半是正例，一半是负例
            '''
        return merged_tokens, merged_tokens_ids, seg_ids, attn_mask

    def __len__(self):
        '''
        一半正例,一半负例
        '''
        return len(self.num_samples)*2


    def _count_lines(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            return sum([1 for _ in f])*2

    def build_index(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            offsets = [0]
            while f.readline():
                offsets.append(f.tell())
        return offsets
    
    def mask_the_sentence(self, tokenized_sentence):
        '''
        15% to process:
            80% -> mask
            10% -> replace
            10% -> keep
        '''
        masked_tokenized_sentence = []
        for token in tokenized_sentence:
            pass

    
    def concate_sentences(self, tokens_a, tokens_b, max_len):
        '''
        return:
            tokens 拼接后的 tokens 带有 特殊token
            tokens_ids 词汇表映射后的 tokens
            seg_ids 区分第一句0还是第二句1
            attn_mask (1, max_len) 得到 mask
        '''
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:(max_len - len(tokens_b) - 3)]
            else:
                tokens_b = tokens_b[:(max_len - len(tokens_a) - 3)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        tokens = tokens + ["[PAD]"] * (max_len - len(tokens))
        tokens_ids = self.vocab.numericalize(tokens)
        seg_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        attn_mask = [1] * len(seg_ids) + [0] * (max_len - len(seg_ids))
        return tokens, tokens_ids, seg_ids, attn_mask

    
    def _get_ids(self):
        pass

    def _get_attn_mask(self):
        pass
