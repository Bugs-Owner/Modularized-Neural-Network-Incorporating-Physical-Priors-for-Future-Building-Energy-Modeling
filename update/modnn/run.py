"""Quick start: train, test and check a ModNN model.

    python -m modnn.run path/to/your_data.csv [preset]

The CSV needs a datetime index and the columns temp_room, temp_amb, solar, occ and phvac.
preset: "consistent" (default), "accurate" or "strict". Outputs go to ./modnn_output.
"""
import random
import sys

import numpy as np
import torch

from modnn import Mod, get_config


def main(datapath, preset="consistent", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args = get_config({"datapath": datapath}, preset=preset)
    mod = Mod(args=args)
    mod.data_ready()
    mod.train()
    mod.load()
    mod.test()
    mod.prediction_show()
    mod.check()
    mod.dynamiccheck()
    mod.check_show()
    return mod


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
