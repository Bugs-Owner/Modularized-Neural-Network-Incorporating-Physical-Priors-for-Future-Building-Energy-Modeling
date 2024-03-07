from Prediction import ddpred
import numpy as np
import torch
import os

for trainday in [30]:
    print(trainday)
    ddp = ddpred()
    path=r"/home/zjiang19/Documents/GitHub/Dynamic/DynamicPrediction/Temperature_SInglezone/EnergyPlus/EP_Training_Data/Single/Train_0_Denver_current_TMY_Cooling.csv"
    ddp.data_ready(path=path,
                   enLen=8,
                   deLen=96,
                   startday=182,
                   trainday=trainday,
                   testday=7,
                   resolution=15,
                   training_batch=512,
                   tar='single_temp')
    def paramethers():
        para = {}
        para["encoLen"], para["decoLen"] = 8, 96
        para["encoder_external_in"], para["encoder_external_h"], para["encoder_external_out"] = 5, 18, 2
        para["encoder_internal_in"], para["encoder_internal_h"], para["encoder_internal_out"] = 4, 16, 2
        para["encoder_hvac_in"], para["encoder_hvac_h"], para["encoder_hvac_out"] = 2, 12, 5

        para["decoder_external_in"], para["decoder_external_h"], para["decoder_external_out"] = 5, 18, 2
        para["decoder_internal_in"], para["decoder_internal_h"], para["decoder_internal_out"] = 4, 16, 2
        para["decoder_hvac_in"], para["decoder_hvac_h"], para["decoder_hvac_out"] = 2, 12, 5

        para["En_out_insize"], para["En_out_h1"], para["En_out_h2"], para["En_out_outsize"] = (
                    para["encoder_external_out"] + para["encoder_internal_out"] +
                    para["encoder_hvac_out"]), 24, 24, 1
        para["De_out_insize"], para["De_out_h1"], para["De_out_h2"], para["De_out_outsize"] = (
                    para["decoder_external_out"] + para["decoder_internal_out"] +
                    para["decoder_hvac_out"]), 24, 24, 1
        para["Num_of_zone"]=1
        return para

    para=paramethers()
    para["lr"], para["epochs"]= 0.001, 200
    ddp.Singletrain(para)
    ddp.Singleload(para)
    ddp.Singletest()
    ddp.Single_Temp_show()
    ddp.Single_save()

    # folder_path = "PretrainedClass"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # save_dic = {'ddpc': ddp,}
    # savename = '{}_ddpc.pt'.format(str(trainday))
    # savefile = os.path.join(folder_path, savename)
    # torch.save(save_dic, savefile)
