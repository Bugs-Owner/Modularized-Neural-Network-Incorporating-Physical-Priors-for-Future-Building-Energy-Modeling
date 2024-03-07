from Prediction import ddpred

#Feel free to adjust these parameters
k=20000
for trainday in [30]:
    for enco in [4]:
        for deco in [96]:
            ddp = ddpred()
            ddp.data_ready(path="/home/zjiang19/Documents/GitHub/Physical-Incorporated-Neural-Network-BEM/Dataset/Real-world-dataset/traindata.csv",
                           enLen=enco,
                           deLen=deco,
                           startday=0,
                           trainday=trainday,
                           testday=14,
                           resolution=15,
                           training_batch=512)

            def paramethers():
                para = {}
                para["encoLen"], para["decoLen"] = enco, deco
                para["encoder_external_in"], para["encoder_external_h"], para["encoder_external_out"] = 5, 18, 4
                para["encoder_internal_in"], para["encoder_internal_h"], para["encoder_internal_out"] = 4, 16, 4
                para["encoder_hvac_in"], para["encoder_hvac_h"], para["encoder_hvac_out"] = 2, 12, 4

                para["decoder_external_in"], para["decoder_external_h"], para["decoder_external_out"] = 5, 18, 4
                para["decoder_internal_in"], para["decoder_internal_h"], para["decoder_internal_out"] = 4, 16, 4
                para["decoder_hvac_in"], para["decoder_hvac_h"], para["decoder_hvac_out"] = 2, 12, 4

                para["En_out_insize"], para["En_out_h1"], para["En_out_h2"], para["En_out_outsize"] = (
                            para["encoder_external_out"] + para["encoder_internal_out"] +
                            para["encoder_hvac_out"]), 24, 24, 1
                para["De_out_insize"], para["De_out_h1"], para["De_out_h2"], para["De_out_outsize"] = (
                            para["decoder_external_out"] + para["decoder_internal_out"] +
                            para["decoder_hvac_out"]), 24, 24, 1
                return para

            para=paramethers()
            para["lr"], para["epochs"]= 0.001, 300
            ddp.train(para)
            ddp.load(para)
            ddp.test()
            ddp.prediction_show()
            ddp.check()
            ddp.check_show(check='HVAC')



