def paras(args):
    para = {}
    para["encoLen"], para["decoLen"] = args.enco, args.deco
    para["lr"], para["epochs"] = args.lr, args.epochs
    para["patience"] = args.patience

    para["encoder_external_in"], para["encoder_external_h"], para["encoder_external_out"] = 5, 12, 3
    para["encoder_internal_in"], para["encoder_internal_h"], para["encoder_internal_out"] = 4, 14, 3
    para["encoder_hvac_in"], para["encoder_hvac_h"], para["encoder_hvac_out"] = 1, 6, 3

    para["decoder_external_in"], para["decoder_external_h"], para["decoder_external_out"] = 5, 12, 3
    para["decoder_internal_in"], para["decoder_internal_h"], para["decoder_internal_out"] = 4, 14, 3
    para["decoder_hvac_in"], para["decoder_hvac_h"], para["decoder_hvac_out"] = 1, 6, 3

    para["En_out_insize"], para["En_out_h1"], para["En_out_h2"], para["En_out_outsize"] = (
            para["encoder_external_out"] + para["encoder_internal_out"] +
            para["encoder_hvac_out"]), 24, 24, 1
    para["De_out_insize"], para["De_out_h1"], para["De_out_h2"], para["De_out_outsize"] = (
            para["decoder_external_out"] + para["decoder_internal_out"] +
            para["decoder_hvac_out"]), 24, 24, 1
    para["En_out_hidden"] = 24
    para["De_out_hidden"] = 24

    return para
