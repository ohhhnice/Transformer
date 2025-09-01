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
from load_data.translation_data.stream_load_data import translation_dataloader_huge
import random


def translate_sentence(model, sentence, zh_vocab, en_vocab, zh_tokenizer, device, tgt_max_len=50, topk=1):
    model.eval()
    tokens = zh_tokenizer(sentence)
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    #tokens = tokens + ["<PAD>"] * (src_max_len - len(tokens))
    src_indices = [zh_vocab.word2idx.get(token, zh_vocab.word2idx["<UNK>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = get_padding_mask(src_tensor, pad_idx=zh_vocab.word2idx["<PAD>"]).to(device)

    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)

    tgt_indices = [en_vocab.word2idx["<SOS>"]]
    for i in range(tgt_max_len-1):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)  # (1, tgt_len)
        tgt_mask = get_tgt_mask(tgt_tensor, pad_idx=en_vocab.word2idx["<PAD>"]).to(device)
        # cross_mask = get_padding_mask(tgt_tensor, pad_idx=zh_vocab.word2idx["<PAD>"]).to(device)
        cross_mask = src_mask

        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, tgt_mask, cross_mask)
            topk_values, topk_indices = torch.topk(output[:, -1, :], k=topk, dim=-1)
            next_token = topk_indices[0, random.randint(0, topk-1)].item()
            # next_token = output.argmax(-1)[:, -1].item()
            tgt_indices.append(next_token)

            if next_token == en_vocab.word2idx["<EOS>"]:
                break

    translated_tokens = [en_vocab.idx2word[idx] for idx in tgt_indices[1:]]  # 去掉<SOS>
    return translated_tokens
    




if __name__ == "__main__":

    SEED = 1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 30

    # 加载模型设置
    print("load model config...")
    model_config_file = './config/transformer_config.json'
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)

    # 加载数据
    # train_dataloader, test_dataloader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer  = translation_dataloader(
    #     batch_size=batch_size,
    #     max_len=model_config['seq_len'], seed=SEED,
    #     json_file = './load_data/translation_data/translation2019zh/translation2019zh_valid.json')

    print("load data...")
    train_dataloader, test_dataloader, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer = translation_dataloader_huge(
        batch_size = batch_size,
        src_max_len=model_config['src_seq_len'],
        tgt_max_len=model_config['tgt_seq_len'],
        train_json_file = './load_data/translation_data/translation2019zh/translation2019zh_train.json',
        val_json_file = './load_data/translation_data/translation2019zh/translation2019zh_valid.json',
        seed=SEED,
        src_filepath_word2idx="./load_data/translation_data/vocab/zh_word2idx.json",
        src_filepath_idx2word="./load_data/translation_data/vocab/zh_idx2word.json",
        tgt_filepath_word2idx="./load_data/translation_data/vocab/en_word2idx.json",
        tgt_filepath_idx2word="./load_data/translation_data/vocab/en_idx2word.json",
        is_train=False
    )

    padding_idx = src_vocab.word2idx["<PAD>"]

    # load model
    print("load model...")
    model = Transformer(encoder_vocab_size = 65472,#len(src_vocab), 
                        decoder_vocab_size = 49003,#len(tgt_vocab), 
                        d_model = model_config['d_model'], 
                        n_heads = model_config['n_heads'], 
                        d_ff = model_config['d_ff'], 
                        encoder_num_layers = model_config['num_encoder_layers'], 
                        decoder_num_layers = model_config['num_decoder_layers'], 
                        encoder_seq_len = model_config['src_seq_len'], 
                        decoder_seq_len = model_config['tgt_seq_len'], 
                        dropout = model_config['dropout']).to(device)

    model.load_state_dict(torch.load('./pth/best_transformer_epoch3.pth', map_location=device))
        
    # 测试翻译
    test_sentences = [
        "你好，你怎么样？",
        "我很好，谢谢你。",
        "你叫什么名字？",
        "我的名字是约翰。",
        "我喜欢学习中文。",
        "这是一个Transformer模型。",
        "我喜欢学英文",
        "总算成功了",
        "你好，你怎么样？你好，你怎么样？你好，你怎么样？你好，你怎么样？",
        "第三章总结了高句丽民俗对周边政权及后世的影响。"
    ]

    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, src_vocab, tgt_vocab, src_tokenizer, device,
            tgt_max_len=model_config['tgt_seq_len'],
            topk=2
        )
        print(f"中文: {sentence}")
        print(f"英文: {' '.join(translation)}")
        print("---")