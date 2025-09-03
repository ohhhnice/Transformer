from transformers import AutoTokenizer
def get_multilingual_tokenizer(model_name="./load_data/translation_data/bert_data_loader/tokenizer/"):
    """
    获取支持中英文的多语言分词器
    "bert-base-multilingual-cased"
    
    model_name选项：
    - bert-base-multilingual-cased: 支持100+语言，包括中英
    - xlm-roberta-base: 更强大的多语言支持
    - hfl/chinese-bert-wwm-ext: 中文优化版，同时兼容英文
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

if __name__ == "__main__":
    tokenizer = get_multilingual_tokenizer()
    text_zh = "你好，世界！"
    text_en = "Hello, world!"
    tokens_zh = tokenizer.tokenize(text_zh)
    tokens_en = tokenizer.tokenize(text_en)
    print("中文分词结果:", tokens_zh)
    print("英文分词结果:", tokens_en)