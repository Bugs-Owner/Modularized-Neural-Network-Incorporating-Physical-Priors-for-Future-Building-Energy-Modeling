from Load_prediction import ddpred
from Para import parameter
import numpy as np



trainday = 180
zones = 'Multi'
modes='Heating'
par = parameter()
par.para_load(zone=zones)
ddp = ddpred()
adj = np.array([[0, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 1, 0]]).astype('float32')
ddp.data_ready(num_zone=5,
               enLen=par.paradic["encoLen"],
               deLen=par.paradic["decoLen"],
               startday=2,
               trainday=trainday,
               testday=180,
               resolution=15,
               training_batch=512,
               adj=adj,
               wea='TMY',
               delta=0,
               city='Denver',
               timespan='current',
               mode=modes)
par.paradic["mode"]=modes
par.paradic["lr"], par.paradic["epochs"] = 0.001, 300
# ddp.train(para=par.paradic)
# ddp.load(para=par.paradic)
# ddp.multitest()
# #ddp.multiplot(feature='phvac')
# ddp.multi_csv_save(feature='phvac')



