from Load_prediction import ddpred
from Para import parameter
import numpy as np



trainday = 180
zones = 'Single'
modes='Heating'
par = parameter()
par.para_load(zone=zones)
ddp = ddpred()
adj = np.array([[1]]).astype('float32')
ddp.data_ready(num_zone=1,
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
par.paradic["lr"],par.paradic["epochs"] = 0.001, 300
#ddp.train(para=par.paradic)
#Test on different task
#ddp.load(para=par.paradic)
#ddp.test()
#ddp.plot(feature='phvac')
# ddp.csv_save(feature='phvac')
#Test on Ehvac task
#ddp.transfer(para=par.paradic)
#ddp.Retrain(feature='Etotal')
ddp.reload(para=par.paradic,feature='Eheat')
#ddp.plot(feature='HVAC')
ddp.csv_save(feature='Eheat')


