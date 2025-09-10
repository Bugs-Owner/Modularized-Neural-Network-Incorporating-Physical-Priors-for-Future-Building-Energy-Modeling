from modnn.utils import Mod
from modnn.Config import _args
import torch
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

args = _args()
Mod = Mod(args=args)
Mod.data_ready()
Mod.train()
Mod.load()
# Mod.load('/home/zjiang19/Documents/GitHub/Physical-Incorporated-Neural-Network-BEM/update/Saved/Eplus/Trained_mdlEnco48_Deco96/PI-modnn_150daysTest_on06-30.pth')
Mod.test()
Mod.prediction_show()
Mod.check()
Mod.dynamiccheck()
Mod.check_show()
# Mod.policy_train()

