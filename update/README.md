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
