import torch
import torch.nn as nn
import math
class Self_Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Self_Attention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.output_LayerNorm = nn.LayerNorm(d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.Dropout1 = nn.Dropout(dropout)
        self.Dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        """
        q:batch_size, seq_len, d_model
        mask: (1, seq_len)
        """
        residual = q
        batch_size, seq_len, d_model = q.size()
        q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(attn, dim=-1)
        scores = self.Dropout1(scores)
        output = torch.matmul(scores, v).transpose(2,1).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)
        output = self.Dropout2(output)
        output = self.output_LayerNorm(output + residual)
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = Self_Attention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.intermediate = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
    def forward(self, x, mask=None):
        attn_output = self.attention(x,x,x,mask)
        output = self.intermediate(attn_output)
        output = self.output(output)
        output = self.LayerNorm(output)
        return output

class Embedding(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, dropout, pad_token_id=0):
        super(Embedding, self).__init__()
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.word_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id) 
        self.token_type_embeddings = nn.Embedding(2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )
    def forward(self, x, token_type_ids):
        seq_length = x.size(1)
        position_ids = self.position_ids[:, 0 : seq_length + 0]
        inputs_embeds = self.word_embeddings(x)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class Bert(nn.Module):
    def __init__(self, d_model, max_len, vocab_size, d_ff, n_heads, num_hidden_layers, dropout=0.1):
        super(Bert, self).__init__()
        self.embeddings = Embedding(d_model, max_len, vocab_size, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout) for _ in range(num_hidden_layers)])
    def forward(self, input_ids, token_type_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)

        encoder_output = embedding_output
        for layer in self.encoder:
            print(type(layer))
            encoder_output = layer(encoder_output, attention_mask)
        return encoder_output

    def load_dict(self, safetensor_path):
        from safetensors.torch import save_file, load_file
        loaded_state = load_file(safetensor_path)
        self.embeddings.LayerNorm.weight = torch.nn.Parameter(loaded_state['bert.embeddings.LayerNorm.gamma'])
        self.embeddings.LayerNorm.bias = torch.nn.Parameter(loaded_state['bert.embeddings.LayerNorm.beta'])
        self.embeddings.position_embeddings.weight = torch.nn.Parameter(loaded_state['bert.embeddings.position_embeddings.weight'])
        self.embeddings.word_embeddings.weight = torch.nn.Parameter(loaded_state['bert.embeddings.word_embeddings.weight'])
        self.embeddings.token_type_embeddings.weight = torch.nn.Parameter(loaded_state['bert.embeddings.token_type_embeddings.weight'])
        self.eval()

input_ids = torch.zeros((2, 128), dtype=torch.long)  # 批次2，序列长度128
attention_mask = torch.ones(2, 128, dtype=torch.float32)        # 无padding，全1掩码
token_type_ids = torch.zeros(2, 128, dtype=torch.long)  
model = Bert(
    d_model=768,
    max_len=512,
    vocab_size=30522,
    d_ff=3072,
    n_heads=12,
    num_hidden_layers=12,
)
encoder_output = model(input_ids, token_type_ids, attention_mask)