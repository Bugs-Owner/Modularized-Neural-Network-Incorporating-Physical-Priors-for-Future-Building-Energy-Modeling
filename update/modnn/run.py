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
Mod.test()
Mod.prediction_show()
Mod.check()
Mod.dynamiccheck()
Mod.check_show()
Mod.grad_check()

