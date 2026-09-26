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

args = get_config({"datapath": "your_data.csv"}, preset="consistent")
model = Mod(args)        # prints a short note on what this setting guarantees
model.data_ready()       # or model.data_ready(df) with a pandas DataFrame
model.train()
model.load()
model.test()
```

or from the command line: `python -m modnn.run your_data.csv consistent`

Scalers, checkpoints, trained models, results and figures are written to `./modnn_output`
(change with `"output_dir"`). The GPU is used when available, otherwise the CPU.

## 🧩 Model presets
Pick a ready-made setting with `preset` (any other override is applied on top):

| preset | design | good for |
| --- | --- | --- |
| `"accurate"` | 1.0.1 (LSTM envelope) | temperature and load forecasting: best accuracy |
| `"consistent"` | 3.0.0 (RNN envelope, sign-constrained) | control, optimization and what-if studies: responses to weather, occupancy and HVAC follow physics |
| `"strict"` | monotone envelope with explicit conduction | applications that require guaranteed physical consistency |

The same switches are available one by one:
* `architecture`: `"v3"` (default) or `"v1"` (first-generation LSTM envelope)
* `ext_input`: `"state"` feeds [T_zone, T_ambient] to the envelope module, `"delta"` feeds T_ambient - T_zone
* `consistency`: `"none"`, `"partial"` or `"strict"` (needs `ext_mdl="RNN"`)

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
Add model presets: "consistent", "accurate" and "strict"

Physical constraints are now applied through `model.apply_constraints()`

New options: "architecture" (v1/v3), "ext_input" (state/delta) and "consistency" (none/partial/strict)

Easier to use: `from modnn import get_config, Mod`, `python -m modnn.run your_data.csv`, outputs in `./modnn_output`
(previously `../`), device defaults to "cuda" with CPU fallback, clear errors for a missing data file or column, scipy added to requirements

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
