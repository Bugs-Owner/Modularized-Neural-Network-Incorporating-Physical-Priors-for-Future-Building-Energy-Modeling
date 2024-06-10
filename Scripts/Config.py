def paras(args):
    para = {}
    para["encoLen"], para["decoLen"] = args.enco, args.deco
    para["lr"], para["epochs"] = args.lr, args.epochs
    para["patience"] = args.patience
    para['HVAC_module'] = args.HVAC_module
    para["encoder_external_in"], para["encoder_external_h"], para["encoder_external_out"] = 5, 15, 5
    para["encoder_internal_in"], para["encoder_internal_h"], para["encoder_internal_out"] = 4, 15, 5
    para["encoder_hvac_in"], para["encoder_hvac_h"], para["encoder_hvac_out"] = 1, 9, 5

    para["decoder_external_in"], para["decoder_external_h"], para["decoder_external_out"] = 5, 15, 5
    para["decoder_internal_in"], para["decoder_internal_h"], para["decoder_internal_out"] = 4, 15, 5
    para["decoder_hvac_in"], para["decoder_hvac_h"], para["decoder_hvac_out"] = 1, 9, 5

    para["En_Nonlinear_out_insize"], para["En_Nonlinear_out_hidden"], para["En_Nonlinear_out_outsize"] = (
        para["encoder_external_out"] + para["encoder_internal_out"], 15, 5)
    para["De_Nonlinear_out_insize"], para["De_Nonlinear_out_hidden"], para["De_Nonlinear_out_outsize"] = (
        para["decoder_external_out"] + para["decoder_internal_out"], 15, 5)

    para["En_out_insize"], para["En_out_h1"], para["En_out_h2"], para["En_out_outsize"] = (
            para["En_Nonlinear_out_outsize"] +
            para["encoder_hvac_out"]), 23, 25, 1
    para["De_out_insize"], para["De_out_h1"], para["De_out_h2"], para["De_out_outsize"] = (
            para["De_Nonlinear_out_outsize"] +
            para["decoder_hvac_out"]), 23, 25, 1

    para["En_out_insize_relu"], para["En_out_hidden_relu"], para["En_out_outsize_relu"] = (
            para["encoder_external_out"] + para["encoder_internal_out"] + para["encoder_hvac_out"]), 24, 1
    para["De_out_insize_relu"], para["De_out_hidden_relu"], para["De_out_outsize_relu"] = (
            para["decoder_external_out"] + para["decoder_internal_out"] + para["decoder_hvac_out"]), 24, 1

    para["SolNN_encoder_insize"], para["SolNN_encoder_hidden"], para["SolNN_decoder_outsize"] = 10, 24, 1
    para["SolNN_decoder_insize"], para["SolNN_decoder_hidden"], para["SolNN_decoder_outsize"] = 9, 24, 1

    para["En_out_hidden"] = 24
    para["De_out_hidden"] = 24

    return para
