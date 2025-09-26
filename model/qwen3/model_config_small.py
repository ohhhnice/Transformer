from dataclasses import dataclass

@dataclass
class Config:
    d_model = 20
    d_ff = 20
    num_decoder_layers = 3
    num_attention_heads = 2
    num_key_value_heads = 1
    max_seq_len = 10
    num_experts = 4
    num_experts_per_tok = 2
    dropout=0.1
    is_moe_mlp=True
    
    vocab_size = 10
    padding_idx = 0

    batch_size = 2
    seq_len = 10

    learning_rate=0.001

