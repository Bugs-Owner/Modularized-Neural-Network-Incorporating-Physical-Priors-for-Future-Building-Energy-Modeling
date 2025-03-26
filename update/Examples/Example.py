from ModNN.Config import get_config
from ModNN.utils import Mod
from ModNN.Dataset import _get_ModNN_input

# ========================================================
# 🏗️ Step 1: Create Input Data for ModNN (if you don’t have a dataset)
# ========================================================
"""
ModNN expects a CSV or DataFrame with these 5 required columns:
  - temp_room : Room temperature [°F]
  - temp_amb  : Outdoor temperature [°F]
  - solar     : Solar radiation [W/m²]
  - occ       : Occupancy (0–1 or people count)
  - phvac     : HVAC power [W]

Use the helper below if you want to generate a ModNN-ready DataFrame.
"""

n_steps = 96  # One day of 15-minute data

occupancy = [1] * n_steps
hvac = np.linspace(0, 2000, n_steps)
temp_amb = [85] * n_steps
solar = np.sin(np.linspace(0, np.pi, n_steps)) * 500
temp_room = [72] * n_steps

df = _get_ModNN_input(
    start_time="2023-07-01 00:00",
    timestep_minutes=15,
    occupancy=occupancy,
    hvac=hvac,
    temp_amb=temp_amb,
    solar=solar,
    temp_room=temp_room
)

# Save to CSV (optional)
df.to_csv("example_input.csv")

# ========================================================
# ⚙️ Step 2: Configure Model Parameters
# ========================================================

args = get_config({
    "datapath": "example_input.csv",
    "device": "cpu",
    "trainday": 1,
    "testday": 1,
    "para": {
        "epochs": 5  # keep short for demo
    }
})

# Optional: print adjustable parameters
# print_help()


args = get_config({
    "device": "cuda:0",
    "datapath": "/home/zjiang19/Documents/GitHub/Physical-Incorporated-Neural-Network-BEM/update/Dataset/EPlus.csv",
    "para": {"epochs": 100}
})
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
