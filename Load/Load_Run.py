from Load_prediction import ddpred
from Para import parameter
import numpy as np



trainday = 30+30
zones = 'Single'
modes='Cooling'
par = parameter()
par.para_load(zone=zones)
ddp = ddpred()
adj = np.array([[1]]).astype('float32')
ddp.data_ready(num_zone=1,
               enLen=par.paradic["encoLen"],
               deLen=par.paradic["decoLen"],
               startday=182-30,
               trainday=trainday,
               testday=30,
               resolution=15,
               training_batch=512,
               adj=adj,
               wea='TMY',
               delta=0,
               city='Denver',
               timespan='current',
               mode=modes)
par.paradic["mode"]=modes
par.paradic["lr"],par.paradic["epochs"] = 0.001, 300
ddp.train(para=par.paradic)
#Test on different task
ddp.load(para=par.paradic)
ddp.test()
ddp.plot(feature='phvac')
# ddp.csv_save(feature='phvac')
#Test on Ehvac task
# par.paradic["lr"],par.paradic["epochs"] = 0.001, 200
# ddp.transfer(para=par.paradic)
# ddp.Retrain(feature='Etotal')
# ddp.reload(para=par.paradic, feature='Ecool')
#ddp.plot(feature='Etotal')
#ddp.csv_save(feature='Ecool')


