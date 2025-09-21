from transformers import PreTrainedTokenizer
import tiktoken

class Qwen3Tokenizer(PreTrainedTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使用tiktoken作为后端
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    def vocab_size(self):
        return self.encoder.n_vocab
        
    def _tokenize(self, text):
        return self.encoder.encode(text)
    
    def _convert_token_to_id(self, token):
        return token
        
    def _convert_id_to_token(self, index):
        return self.encoder.decode([index])
    
    def decode(self, token_ids, **kwargs):
        return self.encoder.decode(token_ids)
    
    def encode(self, text, **kwargs):
        return self.encoder.encode(text)