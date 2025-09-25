from .get_tokenizer import get_multilingual_tokenizer
from .get_vocab import Vocab
from .prepare_data import TranslationDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def translation_dataloader(batch_size, max_len, tokenizer, vocab, padding_idx, train_json_file, valid_json_file):
    '''
    Args:
        batch_size: int, batch size for dataloader
        max_len: int, max length for padding
        tokenizer: tokenizer function
        vocab: Vocab object
        json_file: str, path to json file containing translation data
    '''
    train_dataset = TranslationDataset(file_path = train_json_file, 
                                        max_len = max_len, 
                                        tokenizer = tokenizer, 
                                        vocab = vocab,
                                        padding_idx = padding_idx)
    val_dataset = TranslationDataset(file_path = valid_json_file, 
                                        max_len = max_len, 
                                        tokenizer = tokenizer, 
                                        vocab = vocab,
                                        padding_idx = padding_idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader