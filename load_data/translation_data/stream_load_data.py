from load_data.translation_data.tokenizer import get_tokenizers
from load_data.translation_data.prepare_data import Vocabulary
from load_data.translation_data.prepare_data import TranslationDatasetHuge
from torch.utils.data import DataLoader
import json
from tqdm import tqdm



def translation_dataloader_huge(batch_size, src_max_len, tgt_max_len,
                                train_json_file, val_json_file, seed=5, 
                                src_filepath_word2idx=None, src_filepath_idx2word=None,
                                tgt_filepath_word2idx=None, tgt_filepath_idx2word=None,
                                is_train=True):
    zh_tokenizer, en_tokenizer = get_tokenizers()

    zh_vocab, en_vocab = Vocabulary(), Vocabulary()

    if src_filepath_word2idx is not None and src_filepath_idx2word is not None and \
       tgt_filepath_word2idx is not None and tgt_filepath_idx2word is not None:
        print("Loaded vocabularies from files.")
        zh_vocab.load_from_file(src_filepath_word2idx, src_filepath_idx2word)
        en_vocab.load_from_file(tgt_filepath_word2idx, tgt_filepath_idx2word)
        print(f"Chinese Vocabulary Size: {len(zh_vocab)}")
        print(f"English Vocabulary Size: {len(en_vocab)}")
        
        if not is_train:
            return None, None, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer
        # 直接返回数据加载器
        train_data = TranslationDatasetHuge(train_json_file, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer, 
                                            src_max_len=src_max_len, tgt_max_len=tgt_max_len)
        test_data = TranslationDatasetHuge(val_json_file, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer,
                                            src_max_len=src_max_len, tgt_max_len=tgt_max_len)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer
    
    # 构建词汇表
    train_lines, val_lines = 0, 0
    with open(train_json_file, 'r', encoding='utf-8') as f:
        for _ in f:
            train_lines += 1
    with open(val_json_file, 'r', encoding='utf-8') as f:
        for _ in f:
            val_lines += 1

    with open(train_json_file, "r", encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            zh_sentence, en_sentence = item['chinese'], item['english']
            zh_tokenized, en_tokenized = zh_tokenizer(zh_sentence), en_tokenizer(en_sentence)
            zh_vocab.add_sentence(zh_tokenized)
            en_vocab.add_sentence(en_tokenized)
            if (idx + 1) % 10000 == 0 or (idx + 1) == train_lines:
                print(f"Processed {idx + 1}/{train_lines} lines for training vocab")
    with open(val_json_file, "r", encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            zh_sentence, en_sentence = item['chinese'], item['english']
            zh_tokenized, en_tokenized = zh_tokenizer(zh_sentence), en_tokenizer(en_sentence)
            zh_vocab.add_sentence(zh_tokenized)
            en_vocab.add_sentence(en_tokenized)
            if (idx + 1) % 10000 == 0 or (idx + 1) == val_lines:
                print(f"Processed {idx + 1}/{val_lines} lines for validation vocab")

    zh_vocab.build()    # 入词汇表
    en_vocab.build()    # 入词汇表
    print(f"Chinese Vocabulary Size: {len(zh_vocab)}")
    print(f"English Vocabulary Size: {len(en_vocab)}")
    
    
    # train_data = TranslationDataset(train_data, zh_vocab, en_vocab, max_len=max_len)
    # test_data = TranslationDataset(test_data, zh_vocab, en_vocab, max_len=max_len)
    train_data = TranslationDatasetHuge(train_json_file, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer,
                                        src_max_len=src_max_len, tgt_max_len=tgt_max_len)
    test_data = TranslationDatasetHuge(val_json_file, zh_vocab, en_vocab, zh_tokenizer, en_tokenizer,
                                        src_max_len=src_max_len, tgt_max_len=tgt_max_len)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    src_vocab, tgt_vocab = zh_vocab, en_vocab
    src_tokenizer, tgt_tokenizer = zh_tokenizer, en_tokenizer
    return train_dataloader, test_dataloader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer