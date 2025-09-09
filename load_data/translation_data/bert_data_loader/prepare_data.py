from torch.utils.data import Dataset
import json
import random
import torch

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
        nsp_label = 1 if idx % 2 == 0 else 0
        idx = idx//2
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            json_line = json.loads(line)
            zh_sentence, en_sentence = json_line["chinese"], json_line["english"]
            if nsp_label == 0:
                random_idx = random.randint(0, self.num_samples//2 - 1)
                while random_idx == idx:
                    random_idx = random.randint(0, self.num_samples//2 - 1)
                f.seek(self.offsets[random_idx])
                line_neg = f.readline().strip()
                json_line_neg = json.loads(line_neg)
                en_sentence = json_line_neg["english"]
            zh_sentence_tokenized, en_sentence_tokenized = self.tokenizer(zh_sentence), self.tokenizer(en_sentence)
            merged_tokens_ids, seg_ids, attn_mask, mask_labels = self.concate_sentences(
                tokens_a=zh_sentence_tokenized, 
                tokens_b=en_sentence_tokenized, 
                max_len=self.max_len)
        # return merged_tokens_ids, seg_ids, attn_mask, mask_labels, [nsp_label]*100
        return {"merged_tokens_ids": torch.tensor(merged_tokens_ids), 
                "seg_ids": torch.tensor(seg_ids), 
                "attn_mask": torch.tensor(attn_mask).unsqueeze(0).unsqueeze(1), 
                "mlm_labels": torch.tensor(mask_labels), 
                "nsp_label": torch.tensor([nsp_label])}

    def __len__(self):
        return self.num_samples


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
        return:
            masked_tokenized_sentence
            mask_labels (1 for masked, 0 for not masked)
        '''
        masked_tokenized_sentence = []
        mask_labels = []
        for token in tokenized_sentence:
            if random.random() < 0.15:
                prob = random.random()
                if prob < 0.8:
                    masked_tokenized_sentence.append("[MASK]")
                elif prob < 0.9:
                    random_token = random.choice(list(self.vocab.word2idx.keys())[5:]) # 去掉特殊符号
                    masked_tokenized_sentence.append(random_token)
                else:
                    masked_tokenized_sentence.append(token)
                mask_labels.extend(self.vocab.numericalize([token]))
            else:
                masked_tokenized_sentence.append(token)
                mask_labels.append(self.padding_idx)
        return masked_tokenized_sentence, mask_labels

    
    def concate_sentences(self, tokens_a, tokens_b, max_len):
        '''
        return:
            tokens 拼接后的 tokens 带有 特殊token
            tokens_ids 词汇表映射后的 tokens
            seg_ids 区分第一句0还是第二句1
            attn_mask (1, max_len) 得到 mask
        '''
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            tokens_a, tokens_b = tokens_a[:int(max_len*0.8)], tokens_b[:int(max_len*0.8)]
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:(max_len - len(tokens_b) - 3)]
            else:
                tokens_b = tokens_b[:(max_len - len(tokens_a) - 3)]
        masked_tokens_a, mask_labels_token_a = self.mask_the_sentence(tokens_a)
        masked_tokens_b, mask_labels_token_b = self.mask_the_sentence(tokens_b)
        masked_tokens = ["[CLS]"] + masked_tokens_a + ["[SEP]"] + masked_tokens_b + ["[SEP]"]
        padding_tokens = masked_tokens + ["[PAD]"] * (max_len - len(masked_tokens))

        mask_labels = [self.padding_idx] + mask_labels_token_a + [self.padding_idx] + mask_labels_token_b + [self.padding_idx]
        mask_labels = mask_labels + [self.padding_idx] * (max_len - len(mask_labels))

        tokens_ids = self.vocab.numericalize(padding_tokens)
        seg_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        attn_mask = [1] * len(seg_ids) + [0] * (max_len - len(seg_ids))
        seg_ids = seg_ids + [0] * (max_len - len(seg_ids))
        return tokens_ids, seg_ids, attn_mask, mask_labels
