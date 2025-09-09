import os
import sys
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from transformers import BertTokenizer, BertModel
model_path = "./model/bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
text = "I like [MASK] world! [SEP] I disklike [MASK] world!"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
output_idx = torch.argmax(output.last_hidden_state, dim=-1).view(-1)

vocab = tokenizer.convert_ids_to_tokens(output_idx)
print(output.last_hidden_state.shape)

input_ids = torch.zeros((2, 128), dtype=torch.long)  # 批次2，序列长度128
attention_mask = torch.ones(2, 128, dtype=torch.float32)        # 无padding，全1掩码
token_type_ids = torch.zeros(2, 128, dtype=torch.long)          # 单段落，全0
sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
output = model(input_ids, attention_mask, token_type_ids)
# print(sequence_output)
print(output.last_hidden_state[0][0][:2])
print(sequence_output[0][0][:2])