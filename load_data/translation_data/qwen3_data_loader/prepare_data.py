from torch.utils.data import Dataset
import json
import random
import torch
from model.qwen3.get_mask import generate_qwen3_mask

class TranslationDataset(Dataset):
    def __init__(self, file_path, max_len, tokenizer, vocab, padding_idx):
        self.file_path = file_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.padding_idx = padding_idx

        self.num_samples = self._count_lines()
        self.offsets = self.build_index()

    def __getitem__(self, idx):
        '''
        idx: 奇数正例，偶数负例
        return:
            merged_tokens_ids, seg_ids, attn_mask, mask_labels, nsp_label
        '''
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            json_line = json.loads(line)
            zh_sentence, en_sentence = json_line["chinese"], json_line["english"]
            zh_sentence_tokenized, en_sentence_tokenized = self.tokenizer(zh_sentence), self.tokenizer(en_sentence)
            input_ids, label_ids, attn_mask = self.concate_sentences(
                tokens_a=zh_sentence_tokenized, 
                tokens_b=en_sentence_tokenized, 
                max_len=self.max_len)
        # return merged_tokens_ids, seg_ids, attn_mask, mask_labels, [nsp_label]*100
        return {"input_ids": input_ids, 
                "label_ids": label_ids, 
                "attn_mask": attn_mask}

    def __len__(self):
        return self.num_samples

    def _count_lines(self):
        with open(self.file_path, "r", encoding='utf-8') as f:
            return sum([1 for _ in f])

    def build_index(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            offsets = [0]
            while f.readline():
                offsets.append(f.tell())
        return offsets

    def concate_sentences(self, tokens_a, tokens_b, max_len):
        '''
        return:
            tokens 拼接后的 tokens 带有 特殊token
            tokens_ids 词汇表映射后的 tokens
            seg_ids 区分第一句0还是第二句1
            attn_mask (1, max_len) 得到 mask

            CLS 对应 <|im_start|>， SEP 对应 <|im_end|>
        '''
        vaild_len = max_len - 8
        if len(tokens_a)>(vaild_len//2):
            tokens_a = tokens_a[:vaild_len//2]
        if len(tokens_b)>(vaild_len//2):
            tokens_b = tokens_b[:vaild_len//2]
        merged_tokens = ["[CLS]"] + ["User"] + ["[CLS]"] + tokens_a + ["[SEP]"] + ["[CLS]"] + ["Assistant"] + ["[CLS]"] + tokens_b + ["[SEP]"]
        merged_tokens = merged_tokens + ["[PAD]"]*(max_len-len(merged_tokens))
        merged_tokens_ids = torch.tensor(self.vocab.numericalize(merged_tokens))
        input_ids, label_ids = merged_tokens_ids[:-1], merged_tokens_ids[1:]
        label_ids = torch.tensor([num for num in label_ids], dtype=label_ids.dtype)
        label_ids[:6+len(tokens_a)] = self.padding_idx

        mask = generate_qwen3_mask(input_ids.unsqueeze(0), pad_token_id=self.padding_idx)
        return input_ids, label_ids, mask
