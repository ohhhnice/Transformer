import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from load_data.translation_data.tokenizer import get_tokenizers
from load_data.translation_data.prepare_data import Vocabulary, TranslationDataset, get_data
from utils.set_seed import set_all_seeds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.transformer import Transformer
from torch import nn
from utils.mask_fun import get_padding_mask, get_tgt_mask
import json
from utils.train_function import train_epoch, evaluate_epoch
from load_data.translation_data.get_dataloader import translation_dataloader


def translate_sentence(model, sentence, zh_vocab, en_vocab, zh_tokenizer, device, max_len=30):
    model.eval()
    tokens = zh_tokenizer(sentence)
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    tokens = tokens + ["<PAD>"] * (max_len - len(tokens))
    src_indices = [zh_vocab.word2idx.get(token, zh_vocab.word2idx["<UNK>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = get_padding_mask(src_tensor, pad_idx=zh_vocab.word2idx["<PAD>"]).to(device)

    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)

    tgt_indices = [en_vocab.word2idx["<SOS>"]]
    for i in range(max_len-1):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)  # (1, tgt_len)
        tgt_mask = get_tgt_mask(tgt_tensor, pad_idx=en_vocab.word2idx["<PAD>"]).to(device)

        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
            next_token = output.argmax(-1)[:, -1].item()
            tgt_indices.append(next_token)

            if next_token == en_vocab.word2idx["<EOS>"]:
                break

    translated_tokens = [en_vocab.idx2word[idx] for idx in tgt_indices[1:]]  # 去掉<SOS>
    return translated_tokens
    




if __name__ == "__main__":

    SEED = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型设置
    model_config_file = './config/transformer_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    # 加载数据
    train_dataloader, test_dataloader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer  = translation_dataloader(
        max_len=model_config['seq_len'], seed=SEED)
    padding_idx = src_vocab.word2idx["<PAD>"]

    # load model
    model = Transformer(encoder_vocab_size = len(src_vocab), 
                        decoder_vocab_size = len(tgt_vocab), 
                        d_model = model_config['d_model'], 
                        n_heads = model_config['n_heads'], 
                        d_ff = model_config['d_ff'], 
                        encoder_num_layers = model_config['num_encoder_layers'], 
                        decoder_num_layers = model_config['num_decoder_layers'], 
                        encoder_seq_len = model_config['seq_len'], 
                        decoder_seq_len = model_config['seq_len'], 
                        dropout = model_config['dropout']).to(device)

    model.load_state_dict(torch.load('best_transformer.pth'))
        
    # 测试翻译
    test_sentences = [
        "你好，你怎么样？",
        "我很好，谢谢你。",
        "你叫什么名字？",
        "我的名字是约翰。",
        "我喜欢学习中文。",
        "这是一个Transformer模型。"
    ]

    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, src_vocab, tgt_vocab, src_tokenizer, device
        )
        print(f"英文: {sentence}")
        print(f"中文: {' '.join(translation)}")
        print("---")