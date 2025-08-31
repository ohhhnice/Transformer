def cal_time(n_encoder_layer, n_decoder_layer, d_model, src_seq, tgt_seq, dff, batch_size, tgt_vocab, batch, floats, use_ratio = 0.77):
    encoder_layer_floats = 2 * batch_size * d_model * src_seq * (4 * d_model + src_seq + 2 * dff)
    decoder_layer_floats = 2 * batch_size * d_model * tgt_seq * (4 * d_model + tgt_seq + 2 * dff + src_seq) + 2*batch_size*d_model*d_model*(src_seq + tgt_seq)*2
    MLP_floats = 2 * batch_size * d_model * tgt_seq * tgt_vocab

    total_floats = n_encoder_layer * encoder_layer_floats + n_decoder_layer * decoder_layer_floats + MLP_floats

    total_time = total_floats * batch / (floats*use_ratio)

    embedding_param = 110000 * d_model
    total_param = n_encoder_layer * (4 * d_model * d_model + 2 * d_model * dff + 5 * d_model) + \
                  n_decoder_layer * (4 * d_model * d_model + 2 * d_model * dff + 19 * d_model + 4*d_model*d_model) + \
                  tgt_vocab * d_model + embedding_param
    

    print("encoder_layer_floats: {:.2f}G".format(encoder_layer_floats/1e9))
    print("decoder_layer_floats: {:.2f}G".format(decoder_layer_floats/1e9))
    print("MLP_floats: {:.2f}G".format(MLP_floats/1e9))
    print("total_floats: {:.2f}G".format(total_floats/1e9))
    print("total_time: {:.2f}hours".format(total_time/3600*3))
    print("total_param: {:.2f}M".format(total_param/1e6))
    print("bytes: {:.2f}GB".format(total_param*4/1e9))
    return total_time


if __name__ == "__main__":
    cal_time(
        n_encoder_layer=6,
        n_decoder_layer=6,
        d_model=256,
        src_seq=30,
        tgt_seq=50,
        dff=1024,
        batch_size=64,
        tgt_vocab=49003,
        batch=80648,
        floats=22.06e12
    )




