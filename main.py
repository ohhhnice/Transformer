import torch
from utils.tokenizer import get_tokenizers
from utils.prepare_data import Vocabulary, TranslationDataset, get_data
from utils.set_seed import set_all_seeds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.transformer import Transformer
from torch import nn
from utils.mask_fun import get_padding_mask, get_tgt_mask
from train.train_function import train_epoch, evaluate_epoch

def main():
    SEED = 5
    config = {
        'batch_size': 2,
        'd_model': 512,
        'max_len': 30,  # 序列长度
        'num_encode_layer': 2,
        'num_decode_layer': 2,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'padding_idx': 0,
        'num_epochs': 10
    }

    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "mps" if torch.backends.mps.is_available() else "cpu"

    en_tokenizer, zh_tokenizer = get_tokenizers()
    en_sentences, zh_sentences = get_data()
    en_tokenized = [en_tokenizer(sentence) for sentence in en_sentences]
    zh_tokenized = [zh_tokenizer(sentence) for sentence in zh_sentences]
    en_vocab = Vocabulary()
    zh_vocab = Vocabulary()
    en_vocab.build_vocabulary(en_tokenized)
    zh_vocab.build_vocabulary(zh_tokenized)

    train_data, test_data = train_test_split(list(zip(en_tokenized, zh_tokenized)), test_size=0.2, random_state=SEED)
    train_data = TranslationDataset(train_data, en_vocab, zh_vocab, max_len=config['max_len'])
    train_data[0]
    test_data = TranslationDataset(test_data, en_vocab, zh_vocab, max_len=config['max_len'])
    train_dataloaeder = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)


    # load model
    model = Transformer(
        src_vocab_size=len(zh_vocab), 
        tgt_vocab_size=len(en_vocab),
        d_model=config['d_model'],
        num_encode_layer=config['num_encode_layer'],
        num_decode_layer=config['num_decode_layer'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_len'],
        padding_idx=config['padding_idx']).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.word2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, 
                           train_dataloaeder, 
                           criterion, 
                           optimizer, 
                           en_vocab.word2idx['<PAD>'], 
                           device)
        
        val_loss = evaluate_epoch(model, 
                             test_dataloader, 
                             criterion, 
                             en_vocab.word2idx['<PAD>'], 
                             device)
        

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')
            print("保存最佳模型")




if __name__ == "__main__":
    main()