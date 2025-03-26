from Prediction import ddpred
from Para import parameter
import numpy as np
import torch
import os


trainday = 30
target = 'pHVAC_estimation'
zones = 'Single'
par = parameter()
par.get_para(tar=target, zone=zones)
ddp = ddpred()
adj = np.array([[1]]).astype('float32')
ddp.data_ready(num_zone=1,
               enLen=par.paradic["encoLen"],
               deLen=par.paradic["decoLen"],
               startday=180,
               trainday=trainday,
               testday=30,
               resolution=15,
               training_batch=512,
               tar=target,
               adj=adj,
               wea='TMY',
               delta=0,
               city='Denver',
               timespan='current')

# par.paradic["lr"], par.paradic["epochs"], par.paradic["Task"] = 0.001, 300, target
# ddp.train(para=par.paradic, action='train',
#           load_start='07-03', load_end='07-06')

par.paradic["lr"], par.paradic["epochs"], par.paradic["Task"] = 0.001, 200, 'eHVAC_estimation'
ddp.train(para=par.paradic, action='energy',
          load_start='07-30', load_end='08-29')
ddp.ReEnergy()
# ddp.test()
#ddp.plot(feature)#
# ddp.testsave()
#ddp.loadsave()

