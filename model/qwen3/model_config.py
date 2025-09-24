from dataclasses import dataclass

@dataclass
class Config:
    d_model = 10
    d_ff = 40
    num_decoder_layers = 2
    num_attention_heads = 10
    num_key_value_heads = 5
    max_seq_len = 100
    num_experts = 7
    num_experts_per_tok = 3
    dropout=0.1
    is_moe_mlp=True
    
    vocab_size = 100

    batch_size = 2
    seq_len = 20

