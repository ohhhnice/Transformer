if __name__ == "__main__":
    import sys
    import os
    cur_folder = os.path.dirname(__file__)
    parent_folder = os.path.dirname(cur_folder)
    sys.path.insert(0, parent_folder)
    import torch
    from qwen3.model_config_small import Config
    from model.qwen3.qwen3 import Qwen3
    from qwen3.mlp import MoeMlp
    from model.qwen3.get_mask import generate_qwen3_mask
    model_config = Config()

    model = Qwen3(model_config)
    x = torch.randint(0, model_config.vocab_size, (model_config.batch_size, model_config.seq_len), dtype=torch.long)
    x = torch.tensor([[1,2,3,4,5,6,7,0,0,0],[1,2,3,4,5,6,7,8,9,0]])
    mask = generate_qwen3_mask(x, model_config.padding_idx)
    output,loss = model(x, mask)
    print(output.shape)
    print(loss)
    # x = torch.randn(model_config.batch_size, model_config.seq_len, model_config.d_model)
    # mlp = MoeMlp(model_config)
    # print(mlp(x).shape)