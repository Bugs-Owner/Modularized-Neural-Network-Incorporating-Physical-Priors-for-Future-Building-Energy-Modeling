def _paras(**kwargs):
    """
    Return hyperparameters for training.
    Accepts keyword arguments to override default values.
    """
    para = {
        # Internal gain module
        "Int_in": 3, "Int_h": 8, "Int_out": 1,

        # External disturbance module
        "Ext_in": 2, "Ext_h": 16, "Ext_out": 1,

        # Zone module
        "Zone_in": 1, "Zone_h": 12, "Zone_out": 1,

        # HVAC module
        "HVAC_in": 1, "HVAC_out": 1,

        # Look back window
        "window" : 1,
        "diff_alpha": 0.3,
        # LSTM baseline
        "LSTM_h" : 24,

        "current_meas_dim": 7,
        "future_disturbance_dim": 6,
        "horizon": 96,
        "hidden_dim": 256,
        "policy_epochs":200, "policy_lr":0.01,

        # Training hyperparameters
        "lr": 0.01,
        "epochs": 100,
        "patience": 5, #Early stop
    }

    # Allow override from kwargs
    para.update(kwargs)

    return para

def _envelops(**kwargs):
    """
    Return envelops thermal boundary to stablize training.
    Accepts keyword arguments to override default values.
    #TODO: Pre-trained envelop library
    """
    envelop = {"direct_coef": 0.1,
               "abs_wall_coef": 0.1,
               "abs_roof_coef": 0.1,
               "r_opaque_coef": 10,
               "c_opaque_coef": 100,
               "r_transparent_coef": 10,
               "c_zone": 1000,
               # Module assembly
               "n_wall": 1,
               "n_roof": 0,
               "n_window": 1,
               }
    # Allow override from kwargs
    envelop.update(kwargs)

    return envelop


# Ready-made model settings. Pass one to get_config(preset=...) or _args(preset=...); any other
# override you pass is applied on top.
#   "accurate"   first-generation architecture (LSTM envelope, v1): lowest forecast error, but the
#                envelope is unconstrained, so responses to weather are not guaranteed physical
#   "consistent" RNN envelope driven by (T_ambient - T_zone) with sign constraints: heat from outside,
#                sun, occupants and HVAC always pushes zone temperature the physical way (recommended)
#   "strict"     monotone model with explicit conduction: every response keeps its physical sign at
#                every horizon, by construction, at some cost in accuracy
PRESETS = {
    "accurate":   {"architecture": "v1", "ext_mdl": "LSTM",
                   "para": {"Int_h": 18, "Ext_in": 5, "Ext_h": 22}},
    "consistent": {"architecture": "v3", "ext_mdl": "RNN", "ext_input": "delta", "consistency": "partial",
                   "para": {"Int_h": 12, "Ext_h": 10}},
    "strict":     {"architecture": "v3", "ext_mdl": "RNN", "ext_input": "state", "consistency": "strict",
                   "para": {"Int_h": 8, "Ext_h": 16}},
}
# Training budget used by all presets
PRESET_TRAINING = {"lr": 0.01, "epochs": 150, "patience": 25}


def _args(**kwargs):
    """
    Returns model configurations.
    Allows keyword-based overrides.
    """
    preset = kwargs.pop("preset", None)
    para_overrides = kwargs.pop("para", {})
    envelop_overrides = kwargs.pop("envelop", {})
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError("Unknown preset '{}', choose from {}".format(preset, list(PRESETS)))
        chosen = {k: v for k, v in PRESETS[preset].items() if k != "para"}
        kwargs = {**chosen, **kwargs}
        para_overrides = {**PRESET_TRAINING, **PRESETS[preset]["para"], **para_overrides}
    args = {
        "para": _paras(**para_overrides),
        "envelop": _envelops(**envelop_overrides),
        # Paths and device
        #/home/zjiang19/Documents/GitHub/Eplus_ModNN_Compare/dataset/Eplus/EPlus_train_noAC.csv---EPlus_train_AC_off_2month
        # "datapath": "/home/zjiang19/Documents/GitHub/Eplus_ModNN_Compare/dataset/Eplus/EPlus_train_AC_off_2month.csv", #"../Dataset/EPlus.csv",
        # "datapath": "/home/zjiang19/Documents/GitHub/Eplus_ModNN_Compare/dataset/Eplus/EPlus_train_case1.csv",
        # "datapath": "/home/zjiang19/Documents/GitHub/Physical-Incorporated-Neural-Network-BEM/update/403_new_dyn.csv",
        # "datapath": "/home/zjiang19/Documents/GitHub/ModNN-RL-403/dataset/Data_Process/data_coe_update.csv",
        # "datapath":"/home/zjiang19/Documents/GitHub/BEST_OPT/dataset/dataset_1.csv",
        "datapath": None, # path to your CSV, or pass a DataFrame to Mod.data_ready(df)
        "device": "cuda", # falls back to CPU when no GPU is available
        "output_dir": "modnn_output", # scalers, checkpoints, trained models, results and figures go here
        "save_name": "Eplus",

        # Data settings
        "use_data_cleaning": True,
        "tolerance_hours": 1,
        "resolution": 15, # 15 minutes data
        "enLen": 48, # "Kind of warm-up"
        "deLen": 96, # Prediction horizon, 96 is for 24 hours
        "startday": 10, # Training data selection
        "trainday": 180, # Training data selection
        "testday": 1, # Testing data selection
        "training_batch": 1024*1,
        "multi_deLen": [4, 8, 16, 24, 32, 48],
        "envelop_mdl": "datadriven", # We provide "physics" and "datadriven"
                                  # "physics" rely on heatbalance equation, "datadriven" is a blackbox
        "architecture": "v3", # "v3": modular RNN envelope (default); "v1": first-generation LSTM envelope
        "ext_input": "state", # envelope input, "state": [T_zone, T_ambient]; "delta": T_ambient - T_zone
        "consistency": "none", # physical consistency of the envelope module (needs ext_mdl="RNN"):
                               # "none": unconstrained; "partial": envelope gain rises with ambient/solar and
                               # falls with zone temperature; "strict": every response keeps its physical sign
        "ext_mdl": "RNN", # We provide LSTM and RNN module, for RNN, we can apply positive constraint easily
                           # But LSTM has Hadamard product, making this constraint hard to integrate
                           # However, disturbance variables always have similiar distribution, in other word, is this constraint really necessary?
        "plott": 'all', # all: If want to see how model response to max heating/cooling; else: only Tzone prediction
        "modeltype": 'PI-modnn', # We also have "LSTM", "PI-modnn", "PI-modnn|C", "PI-modnn|L", "PI-modnn|LC" for fun
                                 # LSTM is the baseline, |C means no constraints, |L means no loss adjustment
        "scale": 1, # scaling factor for HVAC power
        "temp_unit": "C",
        "scaler_save_name": "ModNN_scaler.pkl", # scaler name you want to save
        "scaler_load": None, # load previous scaler
        "user_defined_minmax": # user defined minmax for data scaling
            {
             "temp": None,  # For example (50, 120) Temperature(°F)
             "flux": None  # For example (-5000, 5000) Power(W)
            },
        #Policy NN Args

        "control_mode": "Both",

    }

    args.update(kwargs)

    return args


def get_config(overrides=None, preset=None):
    """
    Adjust parameters as needed.

    Args:
        overrides (dict): override config like:
            {
                "datapath": "your_path.csv",
                "para": {"epochs": 200, "lr": 0.01},
                "envelop": {"n_wall": 4, "n_roof": 1, "abs_wall_coef": 0.1},
                "device": "cuda",
                ...
            }
        preset (str): "accurate", "consistent" or "strict" (see PRESETS), or None for the defaults
    Returns:
        dict: Final configuration dictionary
    """
    overrides = dict(overrides or {})
    if preset is not None:
        overrides.setdefault("preset", preset)
    return _args(**overrides)

# Short notes printed when a model is built, so users know what each setting does and does not guarantee
NOTES = {
    "v1": ("ModNN v1 (preset 'accurate', as in release 1.0.1): typically the most accurate forecasts.\n"
           "  Physically consistent for HVAC: heating always warms and cooling always cools the zone.\n"
           "  NOT constrained for outdoor temperature, solar or occupancy: a what-if change to those inputs can move\n"
           "  the forecast the wrong way. For control or what-if studies use preset='consistent'."),
    "none": ("ModNN v3 with an unconstrained envelope (release 3.0.x default).\n"
             "  Physically consistent for HVAC and internal gains; NOT for outdoor temperature or solar.\n"
             "  Use preset='consistent' for consistent responses, or preset='accurate' for the lowest error."),
    "partial": ("ModNN v3, preset 'consistent' (envelope as in release 3.0.0, with sign constraints).\n"
                "  Heat from outside, sun, occupants and HVAC pushes the zone temperature the physical way.\n"
                "  Recommended for control, optimisation and what-if studies; slightly less accurate than 'accurate'."),
    "strict": ("ModNN v3, preset 'strict': every response keeps its physical sign at every horizon, by construction.\n"
               "  Guaranteed consistency at some cost in accuracy; 'consistent' is usually the better trade-off."),
}


def describe(args):
    """One-paragraph note on what the chosen configuration guarantees."""
    if args.get("envelop_mdl") == "physics":
        return "ModNN with the RC (physics) envelope."
    if args.get("modeltype") == "LSTM":
        return "LSTM baseline: purely data-driven, no physical constraints."
    if args.get("architecture", "v3") == "v1":
        return NOTES["v1"]
    return NOTES[args.get("consistency", "none")]


def print_help():
    print("\n🔧 Adjustable Parameters:\n")
    print("General Args:")
    print("  datapath        (str)  : Path to the dataset CSV")
    print("  device          (str)  : Device to run the model on (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    print("  resolution      (int)  : Data resolution in minutes")
    print("  enLen           (int)  : Encoder sequence length (timesteps), 1 step is 15 minutes")
    print("  deLen           (int)  : Decoder sequence length, it is also prediction horizon (timesteps), 1 step is 15 minutes")
    print("  startday        (int)  : Start day of the dataset for training")
    print("  trainday        (int)  : Number of training days")
    print("  testday         (int)  : Number of test days")
    print("  training_batch  (int)  : Batch size for training")
    print("  plott           (str)  : 'all' to plot prediction results and checking results, 'others' to plot prediction results only")
    print("  modeltype       (str)  : LSTM, PI-modnn, PI-modnn|C, PI-modnn|L, PI-modnn|LC where LSTM is the baseline, |C means no constraints, |L means no loss adjustment")
    print("  scale           (float): Scaling factor for HVAC power")
    print("\nHyperparameters (args['para']):")
    print("  Int_in, Int_h, Int_out      : Internal module input/hidden/output size")
    print("  Ext_in, Ext_h, Ext_out      : External module input/hidden/output size")
    print("  Zone_in, Zone_out           : Zone module input/output size")
    print("  HVAC_in, HVAC_out           : HVAC module input/output size")
    print("  window                      : Look up window size")
    print("  lr                          : Learning rate")
    print("  epochs                      : Max training epochs")
    print("  patience                    : Early stopping patience")
    print("\nEnvelop Configuration (args['envelop']):")
    print("  n_wall, n_roof, n_window    : Number of envelope modules for each component")
    print("  direct_coef                 : Solar transmittance")
    print("  abs_wall_coef, abs_roof_coef: Solar absorbance for wall and roof ")
    print("  r_opaque_coef, c_opaque_coef: RC value for opaque envelope")
    print("  c_zone                      : C value for space")
    print("\n📝 Use `get_config(overrides)` to modify these settings.\n")


