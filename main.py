import torch
from utils.tokenizer import get_tokenizers
from utils.prepare_data import Vocabulary, get_data

def main():
    config = {
        'batch_size': 2,
        'seq_len': 3,
        'd_model': 512,
        'src_vocab_size': 200,  # 词汇表，包括特殊字符
        'tgt_vocab_size': 200,
        'max_len': 10,  # 序列长度
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    en_tokenizer, zh_tokenizer = get_tokenizers()
    en_sentences, zh_sentences = get_data()
    en_tokenized = [en_tokenizer(sentence) for sentence in en_sentences]
    zh_tokenized = [zh_tokenizer(sentence) for sentence in zh_sentences]
    en_vocab = Vocabulary()
    zh_vocab = Vocabulary()
    en_vocab.build_vocabulary(en_tokenized)
    zh_vocab.build_vocabulary(zh_tokenized)


if __name__ == "__main__":
    main()