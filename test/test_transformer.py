
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import torch
import json
from modules.multi_head_attention import MultiHeadAttention
from modules.position_embedding import PositionEncoding
from modules.feed_forward import FeedForward
from modules.encoder_layer import EncoderLayer
from modules.decoder_layer import DecoderLayer

# 加载模型设置
model_config_file = './config/transformer_config.json'
with open(model_config_file, "r", encoding='utf-8') as f:
    model_config = json.load(f)

# 构建测试数据集
batch_size, seq_len, d_model = 2, model_config['seq_len'], model_config['d_model']
q,k,v = torch.randn((batch_size, seq_len, d_model)), torch.randn((batch_size, seq_len+10, d_model)), torch.randn((batch_size, seq_len+10, d_model))
print(q)

# 多头自注意力
n_heads = model_config['n_heads']
attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
output = attn(q,k,v)
print(output)
print(q.shape, output.shape)

# position encoding test
position = PositionEncoding(d_model)
output = position(q)
print(output)
print(q.shape, output.shape)


# ff
d_ff = model_config["d_ff"]
feed_forward = FeedForward(d_model, d_ff)
output = feed_forward(q)
print(output)
print(q.shape, output.shape)


# encoder_layer
encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
output = encoder_layer(q)
print(output)
print(q.shape, output.shape)

# decoder_layer
decoder_layer = DecoderLayer(d_model, n_heads, d_ff)
output = decoder_layer(q, k)
print(output)
print(q.shape, output.shape)

# encoder
from modules.encoder import Encoder
src_vocab_size = 20
num_encoder_layers = model_config['num_encoder_layers']
encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, num_layers=num_encoder_layers)
src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
output = encoder(src)
print(output)
print(src.shape, output.shape)

# decoder
from modules.decoder import Decoder
tgt_vocab_size = 33
num_decoder_layers = model_config['num_decoder_layers']
decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, num_layers=num_decoder_layers)
tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))
output = decoder(tgt, output)
print(output)
print(tgt.shape, output.shape)

# transformer
from model.transformer import Transformer
transformer = Transformer(encoder, decoder)
output = transformer(src, tgt)
print(output)
print(src.shape, tgt.shape, output.shape)



















