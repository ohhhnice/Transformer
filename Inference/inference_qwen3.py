import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
from utils.set_seed import set_all_seeds
from load_data.translation_data.qwen3_data_loader.get_vocab import Vocab
from load_data.translation_data.qwen3_data_loader.get_tokenizer import get_multilingual_tokenizer

from model.qwen3.model_config import Config
from model.qwen3.qwen3 import Qwen3
import random

def translate_sentence(model, sentence, vocab, tokenizer, device, max_len=50, topk=1):
    model.eval()
    tokens = tokenizer(sentence)
    tokens = ["[CLS]"] + ["User"] + ["[CLS]"] + tokens + ["[SEP]"] + ["[CLS]"] + ["Assistant"] + ["[CLS]"]
    #tokens = tokens + ["<PAD>"] * (src_max_len - len(tokens))
    src_indices = vocab.numericalize(tokens)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # (1, src_len)

    while True:
        if len(src_indices) >= max_len: break
        src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model(src_tensor)
            topk_values, topk_indices = torch.topk(output[:, -1, :], k=topk, dim=-1)
            next_token = topk_indices[0, random.randint(0, topk-1)].item()
            # next_token = output.argmax(-1)[:, -1].item()
            src_indices.append(next_token)

            if next_token == vocab.word2idx.get("[SEP]"):
                break

    translated_tokens = [vocab.idx2word.get(idx, "[UNK]") for idx in src_indices]
    return translated_tokens



def main():
    SEED = 5
    num_epochs = 3000
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2idx_path = "./load_data/translation_data/qwen3_data_loader/vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/qwen3_data_loader/vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/qwen3_data_loader/tokenizer/'


    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载数据
    model_config = Config()
    
    model = Qwen3(model_config).to(device)
    model_pth = "./best_transformer_epoch1.pth"
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
    # test_sentences = [
    #     "hello, how about you?",
    #     "i am fine, thank you.",
    #     "what is your name?",
    #     "my name is john.",
    #     "i like studying chinese.",
    #     "this is a transformer model.",
    #     "i like learning english",
    #     "finally succeeded",
    #     "hello, how are you? hello, how are you? hello, how are you? hello, how are you?",
    #     "chapter three summarizes the influence of goguryeo folklore on neighboring regimes and later generations.",
    #     "my name is liu qing"
    # ]

    print("\n翻译测试:")
    for sentence in test_sentences:
        translation = translate_sentence(
            model, sentence, vocab, tokenizer, device,
            max_len=model_config.max_seq_len,
            topk=1
        )
        print(f"中文: {sentence}")
        print(f"英文: {' '.join(translation)}")
        print("---")


if __name__ == "__main__":
    main()