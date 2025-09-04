from .get_tokenizer import get_multilingual_tokenizer
from .get_vocab import Vocab
from .prepare_data import TranslationDataset

def translation_dataloader(batch_size, max_len, tokenizer, vocab, seed=5, json_file = None):
    dataset = TranslationDataset(file_path = json_file, 
                                max_len = max_len, 
                                tokenizer = tokenizer, 
                                vocab = vocab)
    data = dataset[0]
    return data





if __name__=="__main__":
    tokenizer = get_multilingual_tokenizer()
    vocab = Vocab(min_freq=2)
    train_dataloader, test_dataloader = translation_dataloader(
        batch_size=32, max_len=50, tokenizer=tokenizer, vocab=vocab, json_file="./load_data/translation_data/translation2019zh/translation2019zh_valid.json"
    )
    for batch in train_dataloader:
        print(batch)
        break