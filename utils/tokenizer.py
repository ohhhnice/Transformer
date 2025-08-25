import spacy

def load_tokenizers():
    spacy_en = spacy.load('en_core_web_sm')
    spacy_zh = spacy.load('zh_core_web_sm')
    return spacy_en, spacy_zh

def tokenize_en(text, spacy_en):
    """英文分词，返回小写词列表"""
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text, spacy_zh):
    """中文分词，返回词列表"""
    return [tok.text for tok in spacy_zh.tokenizer(text)]


def get_tokenizers():
    spacy_en, spacy_zh = load_tokenizers()
    return lambda text: tokenize_en(text, spacy_en), lambda text: tokenize_zh(text, spacy_zh)