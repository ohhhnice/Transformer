from dataclasses import dataclass

@dataclass
class Config:
    d_model = 256
    d_ff = 1024
    num_decoder_layers = 3
    num_attention_heads = 8
    num_key_value_heads = 4
    max_seq_len = 200
    num_experts = 8
    num_experts_per_tok = 3
    dropout=0.1
    is_moe_mlp=True
    
    vocab_size = 22403

    batch_size = 32
    seq_len = 100

    learning_rate=0.001

