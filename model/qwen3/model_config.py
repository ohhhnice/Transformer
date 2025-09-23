from dataclasses import dataclass

@dataclass
class Config:
    d_model = 10
    d_ff = 40
    num_attention_heads = 10
    num_key_value_heads = 5
    max_seq_len = 100
    dropout=0.1

    batch_size = 2
    seq_len = 20

