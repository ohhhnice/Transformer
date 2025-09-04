from load_data.translation_data.transformer_data_loader.tokenizer import get_tokenizers
from load_data.translation_data.transformer_data_loader.prepare_data import Vocabulary, get_data, get_data_big
from load_data.translation_data.transformer_data_loader.prepare_data import TranslationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def translation_dataloader(batch_size, max_len, seed=5, json_file = None):
    zh_tokenizer, en_tokenizer = get_tokenizers()
    zh_sentences, en_sentences = get_data()
    if json_file is not None:
        zh_sentences, en_sentences = get_data_big(json_file)
    zh_tokenized = [zh_tokenizer(sentence) for sentence in zh_sentences]
    en_tokenized = [en_tokenizer(sentence) for sentence in en_sentences]
    zh_vocab = Vocabulary()
    en_vocab = Vocabulary()
    zh_vocab.build_vocabulary(zh_tokenized)
    en_vocab.build_vocabulary(en_tokenized)
    
    train_data, test_data = train_test_split(list(zip(zh_tokenized, en_tokenized)), test_size=0.2, random_state=seed)
    
    train_data = TranslationDataset(train_data, zh_vocab, en_vocab, max_len=max_len)
    test_data = TranslationDataset(test_data, zh_vocab, en_vocab, max_len=max_len)
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    src_vocab, tgt_vocab = zh_vocab, en_vocab
    src_tokenizer, tgt_tokenizer = zh_tokenizer, en_tokenizer
    return train_dataloader, test_dataloader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer