import torch
from load_data.tokenizer import get_tokenizers
from load_data.prepare_data import Vocabulary, TranslationDataset, get_data
from utils.set_seed import set_all_seeds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from model.transformer_v1 import Transformer
from torch import nn
from utils.mask_fun import get_padding_mask, get_tgt_mask
from utils.train_function import train_epoch, evaluate_epoch


def translate_sentence(model, sentence, en_vocab, zh_vocab, zh_tokenizer, device, max_len=30):
    model.eval()
    tokens = zh_tokenizer(sentence)
    tokens = ["<SOS>"] + tokens + ["<EOS>"]
    src_indices = [zh_vocab.word2idx.get(token, zh_vocab.word2idx["<UNK>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)
    src_mask = get_padding_mask(src_tensor, pad_idx=zh_vocab.word2idx["<PAD>"]).to(device)

    with torch.no_grad():
        enc_output = model.pos_embed(model.src_embed(src_tensor))
        for encode_layer in model.Encode:
            enc_output = encode_layer(enc_output, src_mask)

    tgt_indices = [en_vocab.word2idx["<SOS>"]]
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)  # (1, tgt_len)
        tgt_mask = get_tgt_mask(tgt_tensor, pad_idx=en_vocab.word2idx["<PAD>"]).to(device)

        with torch.no_grad():
            tgt_enco = model.pos_embed(model.tgt_embed(tgt_tensor))
            for decode_layer in model.Decode:
                tgt_enco = decode_layer(tgt_enco, enc_output, tgt_mask, src_mask)
            output = model.output_linear(tgt_enco)
            next_token = output.argmax(-1)[:, -1].item()
            tgt_indices.append(next_token)

            if next_token == en_vocab.word2idx["<EOS>"]:
                break

    translated_tokens = [en_vocab.idx2word[idx] for idx in tgt_indices[1:]]  # 去掉<SOS>
    return translated_tokens
    




if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
            'batch_size': 30,
            'd_model': 512,
            'max_len': 100,  # 序列长度
            'num_encode_layer': 6,
            'num_decode_layer': 6,
            'n_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1,
            'padding_idx': 0,
            'num_epochs': 10
        }

    en_tokenizer, zh_tokenizer = get_tokenizers()
    en_sentences, zh_sentences = get_data()
    en_tokenized = [en_tokenizer(sentence) for sentence in en_sentences]
    zh_tokenized = [zh_tokenizer(sentence) for sentence in zh_sentences]
    en_vocab = Vocabulary()
    zh_vocab = Vocabulary()
    en_vocab.build_vocabulary(en_tokenized)
    zh_vocab.build_vocabulary(zh_tokenized)


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
            model, sentence, en_vocab, zh_vocab, zh_tokenizer, device
        )
        print(f"英文: {sentence}")
        print(f"中文: {' '.join(translation)}")
        print("---")