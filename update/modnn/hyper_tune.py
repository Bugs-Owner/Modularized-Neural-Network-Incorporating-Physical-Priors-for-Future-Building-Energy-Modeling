import optuna
import pickle
import os
from utils import Mod
from Config import _args

# === Config ===
RESULTS_DIR = "optuna_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

model_types = ['PI-modnn', 'PI-modnn|C', 'PI-modnn|L', 'PI-modnn|LC', 'LSTM']

train_days = [7, 30, 90, 180, 300]
n_trials = 30

summary_log_path = os.path.join(RESULTS_DIR, "summary_log.pkl")
if os.path.exists(summary_log_path):
    with open(summary_log_path, "rb") as f:
        summary_log = pickle.load(f)
else:
    summary_log = {}

summary_log = {k: v for k, v in summary_log.items() if not k.startswith("LSTM")}
with open(summary_log_path, "wb") as f:
    pickle.dump(summary_log, f)

# === Loop over configs ===
for model_type in model_types:
    for trainday in train_days:
        config_id = f"{model_type}_train{trainday}"
        result_file = os.path.join(RESULTS_DIR, f"{config_id}.pkl")
        db_file = os.path.join(RESULTS_DIR, f"{config_id}.db")

        # Load existing trial metrics
        all_results = []
        if os.path.exists(result_file):
            with open(result_file, "rb") as f:
                all_results = pickle.load(f)

        def objective(trial):
            if model_type == 'LSTM':
                LSTM_h = trial.suggest_int("LSTM_h", 4, 30, step=2)
                para = {
                    "LSTM_h": LSTM_h,
                    "lr": 0.01,
                    "epochs": 100,
                    "patience": 5,
                    "window_size": 1
                }
            else:
                Int_h = trial.suggest_int("Int_h", 4, 30, step=2)
                Ext_h = trial.suggest_int("Ext_h", 4, 30, step=2)
                para = {
                    "Int_in": 3, "Int_h": Int_h, "Int_out": 1,
                    "Ext_in": 2, "Ext_h": Ext_h, "Ext_out": 1,
                    "Zone_in": 1, "Zone_h": 1, "Zone_out": 1,
                    "HVAC_in": 1, "HVAC_h": 1, "HVAC_out": 1,
                    "lr": 0.01,
                    "epochs": 35,
                    "patience": 3,
                    "window_size": 1
                }

            startday = 564 - trainday

            args = _args(
                para=para,
                modeltype=model_type,
                startday=startday,
                trainday=trainday,
                testday=31,
                datapath= "/home/zjiang19/Documents/GitHub/Physical-Incorporated-Neural-Network-BEM/update/403_new_dyn.csv"
            )

            try:
                model = Mod(args=args)
                model.data_ready()
                model.train()
                model.load()
                metrics = model.test()
                model.dynamiccheck()
                vio, mae, loss_dic = model.vio_eva()

                summary = metrics['summary']
                mae_dict = metrics['mae']
                mse_dict = metrics['mse']
                mape_dict = metrics['mape']
                r2_dict = metrics['r2']

                r2_value = summary.get('r2', 0)
                if r2_value <= 0:
                    score = summary['MAE'] + 10
                else:
                    score = summary['MAE']

            except Exception as e:
                print(f"[Trial {trial.number}] Error: {e}")
                mae, mae_dict, mse_dict, mape_dict, r2_dict = float('inf'), {}, {}, {}, {}

            result = {
                "trial": trial.number,
                "summary": {
                    "score": score,
                    "RMSE": summary.get("RMSE", None) if summary else None,
                    "MAPE": summary.get("MAPE", None) if summary else None,
                    "MSE": summary.get("MSE", None) if summary else None,
                },
                "metrics": {
                    "mae": mae_dict,
                    "mse": mse_dict,
                    "mape": mape_dict,
                    "r2_dict": r2_dict,
                }
            }
            if 'vio' in locals():
                result["dynamic_eval"] = {
                    "vio": vio,
                    "dyn_mae": mae,
                    "loss_dic": loss_dic,
                }
            if model_type == 'LSTM':
                result["LSTM_h"] = LSTM_h
            else:
                result["Int_h"] = Int_h
                result["Ext_h"] = Ext_h

            all_results.append(result)

            with open(result_file, "wb") as f:
                pickle.dump(all_results, f)

            return score

        print(f"\n[START] Tuning {config_id}")
        study = optuna.create_study(
            direction="minimize",
            study_name=config_id,
            storage=f"sqlite:///{db_file}",
            load_if_exists=True
        )

        if len(study.trials) >= n_trials:
            print(f"[SKIP] Already completed: {config_id}")
            continue

        study.optimize(objective, n_trials=n_trials - len(study.trials))

        best_trial = study.best_trial
        summary_log[config_id] = {
            "best_params": best_trial.params,
            "best_mae": best_trial.value
        }

        # Save summary after each config
        with open(summary_log_path, "wb") as f:
            pickle.dump(summary_log, f)

print("\n✅ [ALL DONE] Tuning complete and saved.")
