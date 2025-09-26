from dataclasses import dataclass

@dataclass
class Config:
    d_model = 128
    d_ff = 512
    num_decoder_layers = 3
    num_attention_heads = 4
    num_key_value_heads = 2
    max_seq_len = 140
    num_experts = 8
    num_experts_per_tok = 2
    dropout=0.1
    is_moe_mlp=True
    
    vocab_size = 31219
    padding_idx = 0

    batch_size = 32
    seq_len = 140

    learning_rate=0.001

