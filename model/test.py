if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from qwen3.model_config import Config
    from qwen3.qwen3_dense import Qwen3Dense
    model_config = Config()

    model = Qwen3Dense(model_config)
    x = torch.randint(0, model_config.vocab_size, (model_config.batch_size, model_config.seq_len), dtype=torch.long)
    model(x)