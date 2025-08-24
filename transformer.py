import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import matplotlib.pyplot as plt
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_encode_layer, num_decode_layer, 
                 n_heads, d_ff=2048, dropout=0.1, max_len=500, padding_idx=0):
        super(Transformer, self).__init__()
        self.src_embed = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=d_model, padding_idx=padding_idx)
        self.tgt_embed = nn.Embedding(num_embeddings=tgt_vocab_size, embedding_dim=d_model, padding_idx=padding_idx)
        self.pos_embed = Position_Embedding(d_model, max_len)
        self.Encode = nn.ModuleList([EncodeBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_encode_layer)])  
        self.Decode = nn.ModuleList([DecodeBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_decode_layer)]) 
        self.output_linear = nn.Linear(d_model, tgt_vocab_size) 
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_enco = self.pos_embed(self.src_embed(src))
        for encode_layer in self.Encode:
            src_enco = encode_layer(src_enco, src_mask)

        tgt_enco = self.pos_embed(self.tgt_embed(tgt))
        for decode_layer in self.Decode:
            tgt_enco = decode_layer(tgt_enco, src_enco, tgt_mask, src_mask)
        output = self.output_linear(tgt_enco)
        return output


class EncodeBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(EncodeBlock, self).__init__()
        self.FeedForward = FeedForward(d_model, d_ff, dropout)
        self.MultiHeadAttention = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x, mask=None):
        '''
        mask: (batch_size, seq_len)
        '''
        x = self.MultiHeadAttention(x, x, x, mask)
        x = self.FeedForward(x)
        # x (batch_size, seq_len, d_model)
        return x
        

class DecodeBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super(DecodeBlock, self).__init__()
        self.FeedForward = FeedForward(d_model, d_ff, dropout)
        self.MultiHeadAttention = MultiHeadAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, enco_output, self_mask=None, cross_mask=None):
        x = self.MultiHeadAttention(x, x, x, self_mask)  # Self-attention
        x = self.MultiHeadAttention(x, enco_output, enco_output, cross_mask) # Cross-attention
        x = self.FeedForward(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model%n_heads==0
        
        self.d_k = d_model//n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        input: q,k,v (batch_size, seq_len, d_model)   
        output: x (batch_size, seq_len, d_model)
        '''
        batch_size, seq_len, d_model = q.size()
        Q = self.W_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k) # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            # mask: list[1/0] (batch_size, n_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V).transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        output = self.dropout(self.W_o(output))
        output = self.layer_norm(output + q)
        return output

class Position_Embedding(nn.Module):
    def __init__(self, d_model, max_len = 500):
        super(Position_Embedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(pos * torch.exp(-torch.arange(0, d_model, 2).unsqueeze(0)/d_model * math.log(10000)))
        pe[:, 1::2] = torch.cos(pos * torch.exp(-torch.arange(1, d_model, 2).unsqueeze(0)/d_model * math.log(10000)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]


def padding_fun(input_li, max_len, padding=0):
    '''
    seq_li: (batch_size, seq_len)
    output: (batch_size, max_len)
    '''
    return torch.stack([torch.cat([li, torch.zeros(max_len-len(li), dtype=int)]) for li in input_li])

def get_padding_mask(src, pad_idx=0):
    # src: (batch_size, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
    return src_mask

def get_tgt_mask(tgt, pad_idx=0):
    # tgt: (batch_size, tgt_len)
    tgt_pad_mask = get_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tri_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
    return tgt_pad_mask & tri_mask

def get_mask(src, tgt, pad_idx=0):
    '''
    src: (batch_size, src_len)
    tgt: (batch_size, src_len)
    '''
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask


if __name__ == "__main__":
    batch_size =  2
    seq_len = 3
    d_model = 512
    src_vocab_size = 200 # 词汇表，包括特殊字符
    tgt_vocab_size = 200 
    max_len = 10 # 序列长度

    # data = torch.randn(batch_size, seq_len, d_model)
    # PE = Position_Embedding(d_model)
    # position = PE(data)
    # plt.figure(figsize=(12,6))
    # plt.imshow(PE.pe[0].detach().numpy())
    # plt.show()

    src_seq_len = torch.randint(1, max_len, (batch_size,)) # 不为0，确保有文字
    tgt_seq_len = torch.randint(1, max_len, (batch_size,))
    src_seq = [torch.randint(1, src_vocab_size, (src_len,)) for src_len in src_seq_len] # 不为0，0是mask token
    tgt_seq = [torch.randint(1, tgt_vocab_size, (tgt_len,)) for tgt_len in tgt_seq_len]
    src_seq_padding = padding_fun(src_seq, max_len) # padding 操作
    tgt_seq_padding = padding_fun(tgt_seq, max_len)

    # src_mask, tgt_mask = get_mask(src_seq_padding, tgt_seq_padding, pad_idx=0) # 获取mask
    src_mask = get_padding_mask(src_seq_padding, pad_idx=0) # 获取mask
    tgt_mask = get_tgt_mask(tgt_seq_padding, pad_idx=0) # 获取mask

    model = Transformer(src_vocab_size, tgt_vocab_size, 
                        d_model=512, 
                        n_heads=8, 
                        num_encode_layer=2, 
                        num_decode_layer=2, 
                        d_ff = 2048,
                        max_len=500,
                        dropout=0.1)
    
    output = model(src=src_seq_padding, tgt=tgt_seq_padding, src_mask=src_mask, tgt_mask=tgt_mask)
    print(src_seq_padding.shape, tgt_seq_padding.shape)
    print(output.shape)


    # mulhead = MultiHeadAttention(d_model=d_model, n_heads=2)
    # res = mulhead(data, data, data)
