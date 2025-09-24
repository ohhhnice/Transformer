if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from qwen3.model_config import Config
    from model.qwen3.qwen3 import Qwen3Dense
    from qwen3.mlp import MoeMlp
    model_config = Config()

    model = Qwen3Dense(model_config)
    x = torch.randint(0, model_config.vocab_size, (model_config.batch_size, model_config.seq_len), dtype=torch.long)
    model(x)
    # x = torch.randn(model_config.batch_size, model_config.seq_len, model_config.d_model)
    # mlp = MoeMlp(model_config)
    # print(mlp(x).shape)