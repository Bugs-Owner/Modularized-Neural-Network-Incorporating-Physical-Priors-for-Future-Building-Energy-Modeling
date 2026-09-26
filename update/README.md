# ModNN

**ModNN** is a Modularized Physics-Informed Neural Network for building energy modeling.

It incorporates with physics-informed model structure, loss function, and model constraints.

---

## 🚀 Installation

You can install the package using pip:

pip install modnn



## 🧠 Example
Please find the online Jupyter notebook for a step-by-step instruction:
https://colab.research.google.com/drive/1A2jt1q53RtxGuaoym6N1PmlKELDPpYFX?usp=sharing


## ⚡ Quick start
Your CSV needs a datetime index and the columns `temp_room`, `temp_amb`, `solar`, `occ` and `phvac`.

```python
from modnn import get_config, Mod

args = get_config({"datapath": "your_data.csv"})
model = Mod(args)
model.data_ready()       # or model.data_ready(df) with a pandas DataFrame
model.train()
model.load()
model.test()
```

or from the command line: `python -m modnn.run your_data.csv`

Scalers, checkpoints, trained models, results and figures are written to `./modnn_output`
(change with `"output_dir"`). The GPU is used when available, otherwise the CPU.

## 🧩 Model options
| option | values | meaning |
| --- | --- | --- |
| `architecture` | `"v3"` (default), `"v1"` | `"v1"` is the first-generation model (LSTM envelope, as in 1.0.1) |
| `ext_input` | `"state"` (default), `"delta"` | envelope module input: [T_zone, T_ambient] or T_ambient - T_zone (as in 3.0.0) |
| `constraints` | any of `"hvac"`, `"internal"`, `"ambient"`, `"solar"` | inputs whose effect on zone temperature must follow physics (default `["hvac", "internal"]`) |

`"ambient"` and `"solar"` constrain the envelope module and need `architecture="v3"` with `ext_mdl="RNN"`.
For example, a model whose responses to HVAC, occupancy, weather and sun all follow physics:

```python
args = get_config({"datapath": "your_data.csv", "ext_input": "delta",
                   "constraints": ["hvac", "internal", "ambient", "solar"]})
```

## 🧠 Update log
# 🧠 [2.0.0] 2025 May 9
To further improve physical consistency, 

I replaced heat transfer module by set of energy balance equations, 

Start from version 2.0.0

# 🧠 [2.0.1] 2025 May 10
Add another parameter: "envelop_mdl", 

Allow user to use the new physics based module or previous data driven module. 

# 🧠 [2.0.2] 2025 May 10
Fix bug due to parameter: "envelop_mdl",

Vectorize calculation,speed improved by ~6 times.

# 🧠 [3.0.0] 2025 June 11
Update datadriven modnn
RC based envelop_mdl really hard to tune on new dataset

# 🧠 [3.0.1] 2025 June 11
Add a step function for one step ahead prediction

# 🧠 [3.0.2] 2025 June 11
Fix bug for step function

# 🧠 [3.0.3] 2025 June 11
Fix bug for step function

# 🧠 [3.0.4] 2025 Sept 10
Didn't work on it for 3 months, just update the latest version
Will use it for BESTOpt building dynamic model

# 🧠 [3.0.5] 2025 Sept 15
3.0.4 CAN-NOT work at all, I mistakenly comment one line and add a new line of code

# 🧠 [3.0.6] 2025 Sept 15
Fix temperature unit conversion issue

# 🧠 [3.0.7] 2025 Sept 15
Fix step model zone module dimension issue

# 🧠 [3.1.0] 2026 Sept 25
New options: "architecture" (v1/v3) and "ext_input" (state/delta); physical constraints applied through `model.apply_constraints()`

Easier to use: `from modnn import get_config, Mod`, `python -m modnn.run your_data.csv`, outputs in `./modnn_output`
(previously `../`), device defaults to "cuda" with CPU fallback, clear errors for a missing data file or column, scipy added to requirements

# 🧠 [3.2.0] 2026 Sept 25
Choose which responses are physically constrained with `"constraints"`: any of "hvac", "internal", "ambient", "solar"

Presets removed

## 🧪 Requirements

    Python 3.7+

    PyTorch

    NumPy

    Pandas

    Matplotlib

    Seaborn

    scikit-learn

    tqdm

---
📬 License

MIT License

🙋‍♂️ Author

Zixin Jiang: 
zjiang19@syr.edu
