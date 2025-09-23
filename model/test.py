if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import torch
    from qwen3.model_config import Config
    from qwen3.decoder_block import DecoderBlock
    model_config = Config()

    decoder_block = DecoderBlock(model_config)
    x = torch.randn(model_config.batch_size, model_config.seq_len, model_config.d_model)
    decoder_block(x)