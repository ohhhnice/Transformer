from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_gate = nn.Linear(config.d_model, config.d_ff)
        self.W_up = nn.Linear(config.d_model, config.d_ff)
        self.W_down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        '''
        input:
            x (b, s, d)
        return:
            output (b, s, d)
        '''
        output = F.silu(self.W_gate(x))*self.W_up(x)
        output = self.W_down(self.dropout(output))
        return output

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from model_config import Config
    model_config = Config()
    mlp = MLP(model_config)
    x = torch.randn((model_config.batch_size, model_config.seq_len, model_config.d_model))
    print(mlp(x))