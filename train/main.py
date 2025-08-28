import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from model.transformer import Transformer
from modules.encoder import Encoder
from modules.decoder import Decoder
from load_data.tokenizer import get_tokenizers
from load_data.prepare_data import Vocabulary, get_data
from utils.set_seed import set_all_seeds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn, optim
from utils.train_function import train_epoch, evaluate_epoch
from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch
import json



def main():
    SEED = 1
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_config_file = './config/transformer_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    en_tokenizer, zh_tokenizer = get_tokenizers()
    en_sentences, zh_sentences = get_data()
    en_tokenized = [en_tokenizer(sentence) for sentence in en_sentences]
    zh_tokenized = [zh_tokenizer(sentence) for sentence in zh_sentences]
    en_vocab = Vocabulary()
    zh_vocab = Vocabulary()
    en_vocab.build_vocabulary(en_tokenized)
    zh_vocab.build_vocabulary(zh_tokenized)

    train_data, test_data = train_test_split(list(zip(en_tokenized, zh_tokenized)), test_size=0.2, random_state=SEED)
    
    from load_data.prepare_data import TranslationDataset
    batch_size = 2
    train_data = TranslationDataset(train_data, en_vocab, zh_vocab, max_len=model_config['seq_len'])
    test_data = TranslationDataset(test_data, en_vocab, zh_vocab, max_len=model_config['seq_len'])
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    # load model
    src_vocab_size = len(zh_vocab)
    tgt_vocab_size = len(en_vocab)
    d_model = model_config['d_model']
    seq_len = model_config['seq_len']
    n_heads = model_config['n_heads']
    d_ff = model_config['d_ff']
    dropout = model_config['dropout']
    num_encoder_layers = model_config['num_encoder_layers']
    num_decoder_layers = model_config['num_decoder_layers']
    
    encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, num_layers=num_encoder_layers, seq_len=seq_len, dropout=dropout)
    decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, num_layers=num_decoder_layers, seq_len=seq_len, dropout=dropout)
    model = Transformer(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, en_vocab.word2idx["<PAD>"], device)
        valid_loss = evaluate_epoch(model, test_dataloader, criterion, en_vocab.word2idx["<PAD>"], device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_transformer.pth')

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val. Loss: {valid_loss:.4f}')


if __name__ == "__main__":
    main()


















