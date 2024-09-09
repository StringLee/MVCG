from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder

    activation = args.activation

    num_features = args.num_features
    num_dec_layers = args.num_dec_layers

    decoder_AS_type = args.decoder_AS_type
    loss_E_S_para = args.loss_E_S_para
    loss_E_A_para = args.loss_E_A_para
    loss_E_Z_para = args.loss_E_Z_para

    loss_D_A_para = args.loss_D_A_para
    loss_D_S_para = args.loss_D_S_para

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        norm=norm,
        decoder_AS_type=decoder_AS_type,
        loss_E_S_para=loss_E_S_para,
        loss_E_A_para=loss_E_A_para,
        loss_E_Z_para=loss_E_Z_para,
        loss_D_A_para=loss_D_A_para,
        loss_D_S_para=loss_D_S_para,
    )
    return model
