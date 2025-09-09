import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import torch
from utils.set_seed import set_all_seeds
from model.bert import Bert
from load_data.translation_data.bert_data_loader.get_dataloader import translation_dataloader
from load_data.translation_data.bert_data_loader.get_vocab import Vocab
from load_data.translation_data.bert_data_loader.get_tokenizer import get_multilingual_tokenizer
from utils.train_function_bert import train_epoch, evaluate_epoch
import random

def translate_sentence(model, sentence, vocab, tokenizer, device, max_len, topk=1):
    model.eval()
    zh_sentence, en_sentence = sentence
    tokens_a, tokens_b = tokenizer(zh_sentence), tokenizer(en_sentence)
    if len(tokens_a) + len(tokens_b) + 3 > max_len:
        tokens_a, tokens_b = tokens_a[:int(max_len*0.8)], tokens_b[:int(max_len*0.8)]
    if len(tokens_a) + len(tokens_b) + 3 > max_len:
        if len(tokens_a) > len(tokens_b):
            tokens_a = tokens_a[:(max_len - len(tokens_b) - 3)]
        else:
            tokens_b = tokens_b[:(max_len - len(tokens_a) - 3)]
    tokens_a[min(len(tokens_a), 2)] = "[MASK]"
    tokens_b[min(len(tokens_b), 2)] = "[MASK]"
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    indices = [vocab.word2idx.get(token, vocab.word2idx["[UNK]"]) for token in tokens]
    #indices = indices + [0] * (max_len - len(indices))
    seg_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)
    #seg_ids = seg_ids + [0] * (max_len - len(seg_ids)) 
    
    # for i in range(max_len-len(tokens)):
    with torch.no_grad():
        input_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)  # (1, src_len)
        seg_ids_tensor = torch.LongTensor(seg_ids).unsqueeze(0).to(device)
        mlm_output, nsp_output = model(input_tensor, segment_ids=seg_ids_tensor)
        predicted_token_ids = torch.argmax(mlm_output, dim=-1)
        # topk_values, topk_indices = torch.topk(mlm_output[:, -1, :], k=topk, dim=-1)
        # next_token = topk_indices[0, random.randint(0, topk-1)].item()
        # indices.append(next_token)
        # seg_ids.append(1)
        # if next_token == vocab.word2idx["[SEP]"]:
        #     break
    predicted_token_ids = predicted_token_ids.view(-1)
    translated_tokens = [vocab.idx2word[idx.item()] for idx in predicted_token_ids]  # 去掉<SOS>
    print(nsp_output)
    return tokens, translated_tokens, torch.argmax(nsp_output, dim=-1)


if __name__ == "__main__":
    SEED = 5
    set_all_seeds(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    word2idx_path = "./load_data/translation_data/bert_data_loader/vocab/word2idx.json"
    idx2word_path = "./load_data/translation_data/bert_data_loader/vocab/idx2word.json"
    tokenizer_config_folder = './load_data/translation_data/bert_data_loader/tokenizer/'
    model_config_file = './config/bert_config.json'
    train_data_path = './load_data/translation_data/translation2019zh/translation2019zh_train_test.json'
    valid_data_path = './load_data/translation_data/translation2019zh/translation2019zh_valid.json'

    # 加载词表
    vocab = Vocab(min_freq=50)
    vocab.init_from_file(
        filepath_word2idx=word2idx_path, 
        filepath_idx2word=idx2word_path
    )
    tokenizer = get_multilingual_tokenizer(tokenizer_config_folder)
    padding_idx = vocab.word2idx["[PAD]"]

    # 加载数据
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    _, valid_dataloader = translation_dataloader(
        batch_size=1, 
        max_len=model_config["max_len"], 
        tokenizer=tokenizer,    
        vocab=vocab, 
        padding_idx=padding_idx, 
        train_json_file=valid_data_path,
        valid_json_file=valid_data_path
    )

    # 加载模型
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    model = Bert(vocab_size=len(vocab),
                 encoder_num_layers=model_config['num_encoder_layers'],
                 d_model=model_config['d_model'],
                 max_len=model_config['max_len'],
                 n_heads=model_config['n_heads'],
                 d_ff=model_config['d_ff'],
                 dropout=model_config['dropout']).to(device)
    model.load_state_dict(torch.load('./tmp_pth/best_transformer_epoch8.pth', map_location=device))
    # model.load_state_dict(torch.load('./pth/bert/best_transformer_epoch2.pth', map_location=device))
        
    # 测试翻译
    test_sentences = [
        ("你好，你怎么样？","hello, how about you?"),
        ("你好，你怎么样？","I like fuck!"),
        ("我很好，谢谢你。","I'm very well, thank you."),
        ("你叫什么名字？","What is your name?"),
        ("我的名字是约翰。","My name is John."),
        ("我喜欢学习中文。","I like studying Chinese."),
        ("这是一个Transformer模型。","This is a Transformer model."),
        ("我喜欢学英文","I like learning English"),
        ("总算成功了","Finally succeeded"),
        # ("你好，你怎么样？你好，你怎么样？你好，你怎么样？你好，你怎么样？","Hello, how are you? Hello, how are you? Hello, how are you? Hello, how are you?"),
        # ("第三章总结了高句丽民俗对周边政权及后世的影响。","Chapter Three summarizes the influence of Goguryeo folklore on neighboring regimes and later generations.")
    ]

    # 加载数据
    with open(model_config_file, "r", encoding='utf-8') as f:
        model_config = json.load(f)
    # 加载模型设置
    model = Bert(vocab_size=len(vocab),
                 encoder_num_layers=model_config['num_encoder_layers'],
                 d_model=model_config['d_model'],
                 max_len=model_config['max_len'],
                 n_heads=model_config['n_heads'],
                 d_ff=model_config['d_ff'],
                 dropout=model_config['dropout']).to(device)
    
    print("\n翻译测试:")
    # for sentence in test_sentences:
    #     mask_tokens, translation, judge = translate_sentence(
    #         model, sentence, vocab, tokenizer, device,
    #         max_len=model_config['max_len'],
    #         topk=1
    #     )
    for idx, data in enumerate(valid_dataloader):
        if idx==10:
            break
        # en, zh: (batch_size, seq_len)
        merged_tokens_ids, seg_ids, attn_mask, mask_labels, nsp_label = data["merged_tokens_ids"].to(device), \
            data["seg_ids"].to(device), data["attn_mask"].to(device), data["mlm_labels"].to(device), data["nsp_label"].to(device)
        mlm_output, nsp_output = model(x=merged_tokens_ids, segment_ids=seg_ids, attn_mask=attn_mask)

        mask_tokens = [vocab.idx2word[idx.item()] for idx in merged_tokens_ids[0]]
        raw_tokens = [mask_tok if label==0 else vocab.idx2word[label.item()] for label, mask_tok in zip(mask_labels[0], mask_tokens)]
        pred_tokens = [vocab.idx2word[idx.item()] for idx in torch.argmax(mlm_output, dim=-1).view(-1)]
        reconstructed_tokens = [pred_tok if mask_tok == "[MASK]" else mask_tok for mask_tok, pred_tok in zip(mask_tokens, pred_tokens)]
        judge= torch.argmax(nsp_output, dim=-1)
        print(f"句子对: {raw_tokens}")
        print(f"mask 句子: {mask_tokens}")
        print(f"预测内容: {reconstructed_tokens}")
        print(f"判断对错: {judge}")
        print("---")