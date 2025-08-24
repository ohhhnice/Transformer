import torch

def main():
    config = {
        'batch_size': 2,
        'seq_len': 3,
        'd_model': 512,
        'src_vocab_size': 200,  # 词汇表，包括特殊字符
        'tgt_vocab_size': 200,
        'max_len': 10,  # 序列长度
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')