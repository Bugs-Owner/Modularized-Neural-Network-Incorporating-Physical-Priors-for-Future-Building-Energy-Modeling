def _paras(**kwargs):
    """
    Return hyperparameters for training.
    Accepts keyword arguments to override default values.
    """
    para = {
        # Internal gain module
        "Int_in": 3, "Int_h": 18, "Int_out": 1,

        # External disturbance module
        "Ext_in": 5, "Ext_h": 22, "Ext_out": 1,

        # Zone module
        "Zone_in": 1, "Zone_out": 1,

        # HVAC module
        "HVAC_in": 1, "HVAC_out": 1,

        # Look back window
        "window" : 1,

        # Training hyperparameters
        "lr": 0.01,
        "epochs": 500,
        "patience": 10, #Early stop
    }

    # Allow override from kwargs
    para.update(kwargs)

    return para


def _args(**kwargs):
    """
    Returns model configurations.
    Allows keyword-based overrides.
    """
    para_overrides = kwargs.pop("para", {})
    args = {
        "para": _paras(**para_overrides),

        # Paths and device
        "datapath": "../Dataset/EPlus.csv",
        "device": "cuda:1",

        # Data settings
        "resolution": 15, # 15 minutes data
        "enLen": 48, # "Kind of warm-up"
        "deLen": 96, # Prediction horizon, 96 is for 24 hours
        "startday": 60, # Training data selection
        "trainday": 90, # Training data selection
        "testday": 1, # Testing data selection
        "training_batch": 1024,
        "plott": 'all', # all: If want to see how model response to max heating/cooling; else: only Tzone prediction
        "modeltype": 'PI-modnn', # We also have "LSTM", "PI-modnn", "PI-modnn|C", "PI-modnn|L", "PI-modnn|LC" for fun
                                 # LSTM is the baseline, |C means no constraints, |L means no loss adjustment
        "scale": 1, # scaling factor for HVAC power
    }

    args.update(kwargs)

    return args


def get_config(overrides=None):
    """
    Adjust parameters as needed.

    Args:
        overrides (dict): override config like:
            {
                "datapath": "your_path.csv",
                "para": {"epochs": 200, "lr": 0.01},
                "device": "cuda:1",
                ...
            }
    Returns:
        dict: Final configuration dictionary
    """
    return _args(**(overrides or {}))

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
    print("\n📝 Use `get_config(overrides)` to modify these settings.\n")


