import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.transformer import Transformer
from utils.set_seed import set_all_seeds
from torch import nn, optim
from utils.train_function import train_epoch, evaluate_epoch
from load_data.translation_data.transformer_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.transformer_data_loader.tokenizer_for_bert import translation_dataloader_huge
from load_data.translation_data.bert_data_loader.get_vocab import Vocab
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from model.bert_base_seq2seq import Bert_Base_Seq2seq
from utils.mask_fun import get_padding_mask, get_tgt_mask
import torch
import json
import random

def translate_sentence(model, sentence, vocab, tokenizer, device, tgt_max_len=50, topk=1):
    model.eval()
    tokens = tokenizer(sentence)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    #tokens = tokens + ["<PAD>"] * (src_max_len - len(tokens))
    src_indices = [vocab.word2idx.get(token, vocab.word2idx["[UNK]"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = get_padding_mask(src_tensor, pad_idx=vocab.word2idx["[PAD]"]).to(device)

    with torch.no_grad():
        sentence_ids = torch.zeros_like(src_tensor).to(device)
        enc_output = model.encoder(x=src_tensor, segment_ids=sentence_ids, attn_mask=src_mask)

    tgt_indices = [vocab.word2idx["[CLS]"]]
    for i in range(tgt_max_len-1):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)  # (1, tgt_len)
        tgt_mask = get_tgt_mask(tgt_tensor, pad_idx=vocab.word2idx["[PAD]"]).to(device)
        # cross_mask = get_padding_mask(tgt_tensor, pad_idx=vocab.word2idx["<PAD>"]).to(device)
        cross_mask = src_mask

        with torch.no_grad():
            output = model.decoder(tgt_tensor, enc_output, tgt_mask, cross_mask)
            topk_values, topk_indices = torch.topk(output[:, -1, :], k=topk, dim=-1)
            next_token = topk_indices[0, random.randint(0, topk-1)].item()
            # next_token = output.argmax(-1)[:, -1].item()
            tgt_indices.append(next_token)

            if next_token == vocab.word2idx["[SEP]"]:
                break

    translated_tokens = [vocab.idx2word[idx] for idx in tgt_indices[0:]]
    return translated_tokens



def main():
    SEED = 5
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2idx_path = "./load_data/translation_data/bert_data_loader/large_vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/bert_data_loader/large_vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/bert_data_loader/tokenizer/'
    model_config_file = './config/bert_config.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'

    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    vocab_size = len(vocab)
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载模型
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    encoder_init_pth='./pth/bert/best_transformer_epoch2.pth'
    # model.load_state_dict(torch.load('./pth/bert/best_transformer_epoch2.pth', map_location=device))

    print(1)
    model = Bert_Base_Seq2seq(
        vocab_size=vocab_size,
        encoder_num_layers=model_config['num_encoder_layers'],
        decoder_num_layers=model_config["decoder_num_layers"],
        d_model=model_config['d_model'],
        max_len=model_config['max_len'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        encoder_init_pth=encoder_init_pth,
        dropout=model_config['dropout']
    ).to(device)
    model_pth = "best_transformer_epoch1.pth"
    model.load_state_dict(torch.load(model_pth))

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
        "第三章总结了高句丽民俗对周边政权及后世的影响。",
        "我叫刘晴"
    ]

    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, vocab, tokenizer, device,
            tgt_max_len=model_config['max_len'],
            topk=1
        )
        print(f"中文: {sentence}")
        print(f"英文: {' '.join(translation)}")
        print("---")


if __name__ == "__main__":
    main()