import torch.autograd.functional as F
from modnn.Dataset import DataCook
import pickle
import os
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from modnn.Models import ModNN_phy, BaseNN, ModNN_data
from modnn.Play import train_model, test_model, check_model, dynamic_check_model, grad_model
import matplotlib
import seaborn as sns
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class Mod:
    def __init__(self, args):
        self.dataset = None
        self.model = None
        self.para = None
        self.args = args
        self.device = torch.device(self.args["device"] if torch.cuda.is_available() else "cpu")


    def data_ready(self):
        print('cook data')
        start_time = time.time()
        DC = DataCook(args=self.args)
        DC.cook()
        self.dataset = DC
        print("--- %s seconds ---" % (time.time() - start_time))

    def train(self):
        if "modnn" in self.args["modeltype"]:
            if self.args["envelop_mdl"] == "physics":
                model = ModNN_phy.ModNN(self.args).to(self.device)
            else:
                model = ModNN_data.ModNN(self.args).to(self.device)
        if self.args["modeltype"] == "LSTM":
            model = BaseNN.Baseline(self.args).to(self.device)
        start_time = time.time()
        print('model training')
        self.model, self.train_log = train_model(model=model,
                                                train_loader=self.dataset.TrainLoader,
                                                valid_loader=self.dataset.ValidLoader,
                                                test_loader =self.dataset.TestLoader,
                                                lr=self.args["para"]['lr'],
                                                epochs=self.args["para"]['epochs'],
                                                patience=self.args["para"]['patience'],
                                                tempscal = self.dataset.scalers['temp'],
                                                fluxscal = self.dataset.scalers['flux'],
                                                enlen = self.args['enLen'],
                                                delen=self.args['deLen'],
                                                rawdf = self.dataset.test_raw_df,
                                                plott = self.args["plott"],
                                                modeltype = self.args["modeltype"],
                                                scale = self.args["scale"],
                                                device=self.device,
                                                ext_mdl = self.args["ext_mdl"])
        print("--- %s seconds ---" % (time.time() - start_time))

        folder_name = ("../Saved/{}/Trained_mdl".format(self.args['save_name']) +
                       'Enco{}_Deco{}'.format(str(self.args['enLen']),str(self.args['deLen'])))
        mdl_name = '{}_{}daysTest_on{}.pth'.format(self.args["modeltype"], str(self.args["trainday"]), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savemodel = os.path.join(folder_name, mdl_name)
        torch.save(model.state_dict(), savemodel)

        folder_name = ("../Saved/{}/Loss".format(self.args['save_name']) +
                       'Enco{}_Deco{}'.format(str(self.args['enLen']), str(self.args['deLen'])))
        loss_name = '{}Loss{}days_Test_on{}.pickle'.format(self.args["modeltype"], str(self.args["trainday"]), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        saveloss = os.path.join(folder_name, loss_name)
        with open(saveloss, 'wb') as f:
            pickle.dump(self.train_log, f)

        # #Loss display
        # def _loss_norm(loss):
        #     loss_arr = np.array(loss)
        #     return np.log(loss_arr)
        #
        # color_paletteZ_ = ["#008744", "#0057e7", "#d62d20", "#ffa700"]
        # fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=300, constrained_layout=True)
        # ax.plot(_loss_norm(self.train_log['train_temp_losses']), label='Temp_mse', color='#f24e4c')
        # ax.plot(_loss_norm(self.train_log['train_diff_losses']), label='Temp_diff', color='#bbe088')
        #
        # # ax.plot(self.train_log['valid_losses'], label='Validation_loss', color='#bbe088')
        #
        # ax.plot(_loss_norm(self.train_log['vio_positive_loss']), label='TRV+', color='#537af5')
        # ax.plot(_loss_norm(self.train_log['vio_negative_loss']), label='TRV-', color='#f26fc4')
        # ax.set_ylabel('Loss_Norm', fontsize=7)
        # ax.set_xlabel('Epochs', fontsize=7)
        # ax.tick_params(axis='y', which='both', labelsize=7, pad=0.7)
        # ax.tick_params(axis='x', which='both', labelsize=7, pad=0.7)
        # ax.set_title(self.args["modeltype"], fontsize=7)
        # # ax.legend(loc='upper right', ncol=1, fontsize=6, frameon=False)
        # # ax1.legend(loc='center', bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7, frameon=False)
        # plt.show()

    def train_valid_loss_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(1.96, 1.96), dpi=300, constrained_layout=True)
        ax.plot(self.train_log['train_losses'], label='Training', color='#F0907F')
        ax.plot(self.train_log['valid_losses'], label='Validation', color='#8B90F5')
        ax.set_ylabel('Loss', fontsize=7)
        ax.set_xlabel('Epochs', fontsize=7)
        ax.tick_params(axis='y', which='both', labelsize=7, pad=0.7)
        ax.tick_params(axis='x', which='both', labelsize=7, pad=0.7)
        # ax.set_ylim(0, 0.1)
        # Creating an inset of the size [width, height] starting at position (x, y)
        # all quantities are in fraction of figure width or height
        axins = inset_axes(ax, width="120%", height="120%", loc='upper right',
                           bbox_to_anchor=(0.5, 0.5, 0.45, 0.45),
                           bbox_transform=ax.transAxes)

        # Plot the same data on the inset
        axins.plot(self.train_log['train_losses'], label='Training_loss', color='#F0907F')
        axins.plot(self.train_log['valid_losses'], label='Validation loss', color='#8B90F5')

        # Specify the limits of your zoom-in area
        x1, x2, y1, y2 = 50, 120, 0.001, 0.008
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        axins.grid(True)
        axins.tick_params(axis='both', which='both', labelsize=5)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.13), ncol=1, fontsize=7, frameon=False)
        plt.show()

    def load(self, mdl_name=None):
        """
        :param mdl_name: load an existing model or load just trained model
        :return: model
        """
        start_time = time.time()
        print("Loading model")
        if "modnn" in self.args["modeltype"]:
            if self.args["envelop_mdl"] == "physics":
                model = ModNN_phy.ModNN(self.args).to(self.device)
            else:
                model = ModNN_data.ModNN(self.args).to(self.device)
        if self.args["modeltype"] == "LSTM":
            model = BaseNN.Baseline(self.args).to(self.device)

        folder_name = ("../Saved/{}/Trained_mdl".format(self.args['save_name']) +
                       'Enco{}_Deco{}'.format(str(self.args['enLen']),str(self.args['deLen'])))
        if mdl_name is None:
            mdl_name = '{}_{}daysTest_on{}.pth'.format(self.args["modeltype"],
                                                       str(self.args["trainday"]),
                                                       self.dataset.test_start)
        else:
            mdl_name=mdl_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        loadmodel = os.path.join(folder_name, mdl_name)
        model.load_state_dict(torch.load(loadmodel))
        model.eval()
        self.model = model
        print("--- %s seconds ---" % (time.time() - start_time))

    def _calculate_metrics(y_true, y_pred):
        """
        Calculate error metrics for temperature predictions.

        Args:
            y_true (np.array): Ground truth temperature values
            y_pred (np.array): Predicted temperature values

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        def MAPE(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def MAE(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        def MSE(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        metrics = {
            'MAPE': MAPE(y_true, y_pred),
            'MAE': MAE(y_true, y_pred),
            'MSE': MSE(y_true, y_pred),
            'RMSE': np.sqrt(MSE(y_true, y_pred))
        }

        return metrics

    def _save_results(self, metrics, test_result, folder_prefix="../Result", is_new_data=False):
        """
        Save test results

        Args:
            metrics (dict): Dictionary of evaluation metrics
            test_result (dict): Dictionary of test results
            folder_prefix (str): Prefix for the folder path
            is_new_data (bool): Whether this is for new data test or regular test
        """
        # Determine the test start date description
        if is_new_data:
            if hasattr(self, 'test_start'):
                data_desc = f"new_data_Test_on{self.test_start}"
            else:
                data_desc = "new_data"
        else:
            data_desc = f"Train_with_{self.args['trainday']}days\\nTest_on{self.dataset.test_start}"

        # Create folder if it doesn't exist
        folder_name = (f"{folder_prefix}/{self.args['save_name']}/{self.args['modeltype']}/" +
                       f'Enco{self.args["enLen"]}_Deco{self.args["deLen"]}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save raw data if available for regular test
        if not is_new_data and hasattr(self, 'dataset') and hasattr(self.dataset, 'test_raw_df'):
            csv_name = f'(raw){data_desc}.csv'
            savecsv = os.path.join(folder_name, csv_name)
            self.dataset.test_raw_df.to_csv(savecsv)

        # Save predictions
        dic_name = f'(pred){data_desc}.pkl'
        savedic = os.path.join(folder_name, dic_name)
        with open(savedic, 'wb') as f:
            pickle.dump(test_result, f)

        # Save metrics
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, dict):  # For per-timestep metrics
                metric_file = f'({metric_name.lower()}){data_desc}.pkl'
                save_path = os.path.join(folder_name, metric_file)
                with open(save_path, 'wb') as f:
                    pickle.dump(metric_values, f)
            else:
                pass

    def test(self, testing_data_path=None):
        """
        Test model performance on either the default testing dataset or a specified testing dataset

        Args:
            testing_data_path (str, optional): Path to a new testing dataset

        Returns:
            dict: Performance metrics
        """
        print('Testing start')
        start_time = time.time()

        if testing_data_path is not None:
            print('Testing on new dataset')

            # Load new testing dataset using original scaler
            new_args = self.args.copy()
            new_args["datapath"] = testing_data_path
            new_args["scaler_load"] = True
            new_args["startday"] = 30
            new_args["trainday"] = 180
            new_args["testday"] = 1
            DC_new = DataCook(args=new_args)

            # Load original scaler
            if hasattr(self, 'dataset') and self.dataset is not None and hasattr(self.dataset, 'scalers'):
                DC_new.scalers = self.dataset.scalers
            DC_new.load_data()
            DC_new.prepare_data_splits()
            test_loader = DC_new._create_dataloader(data=DC_new.testingdf, batch_size=len(DC_new.testingdf), shuffle=False)

            # Run test on the new testing dataset
            to_out, en_out, de_out = test_model(model=self.model,
                                                test_loader=test_loader,
                                                device=self.device,
                                                tempscal=DC_new.scalers['temp'],
                                                enlen=self.args['enLen'])

            is_new_data = True
            self.dataset = DC_new
            self.test_start = DC_new.test_start
            self.test_end = DC_new.test_end
        else:
            # Use the default testing dataset
            to_out, en_out, de_out = test_model(model=self.model,
                                                test_loader=self.dataset.TestLoader,
                                                device=self.device,
                                                tempscal=self.dataset.scalers['temp'],
                                                enlen=self.args['enLen'])

            is_new_data = False
        true_temps = self.dataset.test_raw_df['temp_room'].values
        test_df = self.dataset.test_raw_df
        self.to_out, self.en_out, self.de_out = to_out, en_out, de_out
        test_len = len(de_out)
        pred_len = self.args['deLen']

        # Calculate overall metrics
        test_result = {}
        mae = {}
        mse = {}
        mape = {}

        gdtruth = (true_temps - 32) * 5 / 9 if self.args.get("temp_unit", "F") == "F" else true_temps

        for timestep in range(test_len):
            tem = de_out[timestep]
            tem = (tem - 32) * 5 / 9 if self.args.get("temp_unit", "F") == "F" else tem
            test_result[timestep] = tem

            # Calculate metrics for this timestep
            y_true = gdtruth[timestep: timestep + pred_len].reshape(-1, 1)
            y_pred = tem

            mae[timestep] = np.mean(np.abs(y_true - y_pred))
            mse[timestep] = np.mean((y_true - y_pred) ** 2)
            mape[timestep] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Store per-timestep metrics
        self.mae = mae
        self.mse = mse
        self.mape = mape

        # Calculate overall metrics
        avg_metrics = {
            'MAPE': np.mean(list(mape.values())),
            'MAE': np.mean(list(mae.values())),
            'MSE': np.mean(list(mse.values())),
            'RMSE': np.sqrt(np.mean(list(mse.values())))
        }

        # Print metrics
        print(f"Performance metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Save detailed metrics and results
        metrics = {
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'summary': avg_metrics
        }

        # Save results
        self._save_results(metrics, test_result, is_new_data=is_new_data)

        # Store result for new testing data
        if is_new_data:
            self.new_test_results = {
                'true_temps': true_temps,
                'pred_temps': de_out,
                'metrics': avg_metrics
            }

        print(f"--- {time.time() - start_time:.2f} seconds ---")

        return avg_metrics

    def check(self):
        print('Checking start')
        start_time = time.time()

        (self.outputs_max_denorm, self.outputs_min_denorm, self.outputs_denorm,
         self.cooling_max_denorm, self.cooling_min_denorm) = check_model(model=self.model,
                                                                        test_loader=self.dataset.TestLoader,
                                                                        tempscal=self.dataset.scalers['temp'],
                                                                        hvacscale=self.dataset.scalers['flux'],
                                                                        enlen=self.args['enLen'],
                                                                        scale=self.args["scale"],
                                                                        checkscale = 1,
                                                                        device=self.device
                                                                        )
        print("--- %s seconds ---" % (time.time() - start_time))

    def dynamiccheck(self):
        print('Checking start')
        start_time = time.time()
        self.dynamic_temp, self.dynamic_hvac = dynamic_check_model(model=self.model,
                                                                    test_loader=self.dataset.TestLoader,
                                                                    tempscal=self.dataset.scalers['temp'],
                                                                    hvacscale=self.dataset.scalers['flux'],
                                                                    enlen=self.args['enLen'],
                                                                    scale=self.args["scale"],
                                                                    device=self.device
                                                                    )

        print("--- %s seconds ---" % (time.time() - start_time))

    def vio_eva(self):
        vio = 0
        test_len = len(self.de_out)
        _ = self.dynamic_temp[-4000]
        for u in [-2000, 0, 2000, 4000]:
            for timestep in range(len(self.de_out)):
                td = _[timestep][:test_len - timestep] - self.dynamic_temp[u][timestep][:test_len - timestep]
                vio += np.clip(td, 0, None).sum()
                _ = self.dynamic_temp[u]
        return vio, self.mae, self.train_log

    def prediction_show(self):
        print('Ploting start')
        rawdf = self.dataset.test_raw_df
        test_len = len(self.de_out)
        pred_len = self.args['deLen']
        fig, ax = plt.subplots(1, 1, figsize=(6, 1.2), dpi=300, constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_room'].values[:test_len] - 32) * 5 / 9, '-',
                     linewidth=1, color="#159A9C", label='Measurement')
        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        for timestep in range(test_len):
            tem = self.de_out[timestep][:test_len - timestep]
            tem = (tem - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         tem, '--', linewidth=0.3, color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1], ((self.de_out[0][0]) - 32) * 5 / 9, '--', linewidth=1,
                     color="#EA7E7E", label=self.args["modeltype"])
        ax.text(0.01, 0.7, 'MAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(mae, mape), fontsize=6, color='gray',
                fontweight='bold', transform=ax.transAxes)
        ax.legend([], [], frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.xaxis.set_minor_locator(dates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))

        def custom_date_formatter(x, pos):
            dt = dates.num2date(x)  # Convert number to date
            if pos == 0:  # Show the month only at the beginning
                return dt.strftime('%b\n%d')
            return dt.strftime('%d')  # Show only the day elsewhere

        # Set up x-axis ticks
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))

        # ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        # ax.xaxis.set_major_formatter(dates.DateFormatter('%b\n%d'))
        ax.set_xlabel(None)
        # ax.set_ylim(21,27)
        # ax.set_yticks(range(22,28,2))
        ax.set_ylabel('Temperature[°C]', fontsize=7, fontweight='bold')
        ax.margins(x=0)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7, frameon=False)
        plt.show()

    def check_show(self):
        print('Ploting start')
        rawdf = self.dataset.test_raw_df
        test_len = len(self.de_out)
        pred_len = self.args['deLen']
        fig, axes = plt.subplots(2, 1, figsize=(2.84, 1.92*2), dpi=500, sharex='col', sharey='row', constrained_layout=True)
        axes[0].plot_date(rawdf.index[:test_len], (rawdf['temp_room'].values[:test_len] - 32) * 5 / 9, '-',
                     linewidth=1, color="#159A9C", label='GroundTruth')
        axes[1].plot_date(rawdf.index[:test_len], (rawdf['phvac'].values[:test_len])/1000, '-',
                          linewidth=1, color="#159A9C", label='GroundTruth')
        for timestep in range(test_len):
            tem = self.de_out[timestep][:test_len - timestep]
            tem = (tem - 32) * 5 / 9
            axes[0].plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         tem, '--', linewidth=0.3, color="gray")
        axes[0].plot_date(rawdf.index[0:0 + 1], ((self.de_out[0][0]) - 32) * 5 / 9, '--', linewidth=1,
                     color="gray", label=self.args["modeltype"])

        for timestep in [0]:
            color_list={}
            color_list[4000] = '#d73b29'
            color_list[2000] = '#e76248'
            color_list[0] = '#f89574'
            color_list[-2000] = '#73a2c6'
            # color_list[0.2] = '#4771b2'
            color_list[-4000] = '#00429d'

            for u in [-4000, -2000, 0, 2000, 4000]:
                rand = self.dynamic_temp[u][timestep][:test_len - timestep]
                rand = (rand - 32) * 5 / 9
                axes[0].plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], rand, '--', linewidth=1,
                                  color=color_list[u], label='Phvac:{}kW'.format(u))
                hvac = self.dynamic_hvac[u][timestep][:test_len - timestep]
                axes[1].plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], hvac/1000, '--',
                                  linewidth=1,
                                  color=color_list[u], label='Phvac:{}kW'.format(u))

        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        vio = 0
        _ = self.dynamic_temp[-4000]
        for u in [-2000, 0, 2000, 4000]:
            for timestep in range(len(self.de_out)):
                td = _[timestep][:test_len - timestep] - self.dynamic_temp[u][timestep][:test_len - timestep]
                vio +=np.clip(td, 0, None).sum()
                _ = self.dynamic_temp[u]


        axes[0].text(0.01, 0.8, 'MAE:{:.1f}[°C]\nMAPE:{:.1f}[%]\nVIO:{:.1f}[°C-h]'.format(mae, mape, vio/4), fontsize=7,
                transform=axes[0].transAxes)
        axes[1].tick_params(axis='both', which='minor', labelsize=9)
        axes[1].tick_params(axis='both', which='major', labelsize=9)
        axes[1].xaxis.set_minor_locator(dates.HourLocator(interval=6))
        axes[1].xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))
        axes[1].xaxis.set_major_locator(dates.DayLocator(interval=1))
        axes[1].xaxis.set_major_formatter(dates.DateFormatter('%b%d'))
        # axes[0].set_yticks(np.arange(22, 32, 2))
        # axes[0].set_ylim(21, 31.8)
        axes[0].set_xlabel(None)
        axes[0].set_ylabel('Temperature[°C]', fontsize=9)
        axes[1].set_ylabel('HVAC Power[kW]', fontsize=9)
        axes[1].margins(x=0)
        plt.show()
        folder = '../Saved/Checking_Image/{}_{}'.format(rawdf.index[0].month, rawdf.index[0].day)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plot_name = 'Temp[{} to {}].pdf'.format(self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder, plot_name)
        fig.savefig(saveplot, transparent=True, format='pdf')

        # PLot Temp check for paper
        fig, ax = plt.subplots(1, 1, figsize=(2.84, 1.92), dpi=500, constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_room'].values[:test_len] - 32) * 5 / 9, '-',
                     linewidth=1, color="#159A9C", label='GroundTruth')
        for timestep in range(test_len):
            tem = self.de_out[timestep][:test_len - timestep]
            tem = (tem - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         tem, '--', linewidth=0.5, color="gray")
        ax.plot_date(rawdf.index[0:0 + 1], ((self.de_out[0][0]) - 32) * 5 / 9, '--', linewidth=1,
                     color="gray", label=self.args["modeltype"])

        for timestep in [0]:
            minn = self.outputs_min_denorm[timestep][:test_len - timestep]
            minn = (minn - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], minn, '--', linewidth=1,
                         color="#8163FD", label='Max Cooling')
            maxx = self.outputs_max_denorm[timestep][:test_len - timestep]
            maxx = (maxx - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], maxx, '--', linewidth=1,
                         color="#FF5F5D", label='Max Heating')

        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        ax.text(0.01, 0.8, 'MAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(mae, mape), fontsize=9,
                transform=ax.transAxes)

        vio1, vio2 = 0, 0
        #Temperature response violation
        for timestep in range(len(self.de_out)):
            minn = self.outputs_min_denorm[timestep][:test_len - timestep][0]
            minn = (minn - 32) * 5 / 9
            maxx = self.outputs_max_denorm[timestep][:test_len - timestep][0]
            maxx = (maxx - 32) * 5 / 9
            y_pred = (self.de_out[timestep][:test_len - timestep]- 32) * 5 / 9
            y_pred = y_pred[0]
            vio1 += np.clip((minn - y_pred), 0, None)/4
            vio2 += np.clip((y_pred - maxx), 0, None)/4

        text_red = 'TRV+:{:.1f}[°C-h]'.format(vio2[0])
        text_blue = 'TRV-:{:.1f}[°C-h]'.format(vio1[0])
        if 'modnn' in self.args["modeltype"]:
            pad = 0.53
        else:
            pad = 0.49
        ax.text(pad, 0.15, text_red, fontsize=9, color='red', transform=ax.transAxes)
        ax.text(pad, 0.01, text_blue, fontsize=9, color='blue', transform=ax.transAxes)

        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9, frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.xaxis.set_minor_locator(dates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b%d'))
        ax.set_xlabel(None)
        ax.set_ylabel('Temperature[°C]', fontsize=9)
        ax.margins(x=0)
        # ax.set_yticks(np.arange(20, 40, 4))
        # ax.set_ylim(18, 40)
        plt.show()

    def overall_show(self):
        rawdf = self.dataset.test_raw_df
        test_len = len(self.de_out)
        pred_len = self.dataset.deLen
        if self.args.modeltype == 'Baseline':
            LB = 'LSTM'
        else:
            LB = 'modnn'
        test_len=96

        # PLot Temp check for paper
        fig, ax = plt.subplots(1, 1, figsize=(2.84, 1.92), dpi=500, constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9, '-',
                     linewidth=1, color="#159A9C", label='GroundTruth')
        for timestep in range(test_len):
            tem = self.de_out[timestep][:test_len - timestep]
            tem = (tem - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         tem, '--', linewidth=0.5, color="gray")
        ax.plot_date(rawdf.index[0:0 + 1], ((self.de_out[0][0]) - 32) * 5 / 9, '--', linewidth=1,
                     color="gray", label=LB)

        for timestep in [0]:
            minn = self.outputs_min_denorm[timestep][:test_len - timestep]
            minn = (minn - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], minn, '--', linewidth=1,
                         color="#8163FD", label='Max Cooling')
            maxx = self.outputs_max_denorm[timestep][:test_len - timestep]
            maxx = (maxx - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], maxx, '--', linewidth=1,
                         color="#FF5F5D", label='Max Heating')

        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        ax.text(0.01, 0.8, 'MAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(mae, mape), fontsize=9,
                transform=ax.transAxes)

        vio1, vio2 = 0, 0
        #Temperature response violation
        for timestep in range(len(self.de_out)):
            minn = self.outputs_min_denorm[timestep][:test_len - timestep][0]
            minn = (minn - 32) * 5 / 9
            maxx = self.outputs_max_denorm[timestep][:test_len - timestep][0]
            maxx = (maxx - 32) * 5 / 9
            y_pred = (self.de_out[timestep][:test_len - timestep]- 32) * 5 / 9
            y_pred = y_pred[0]
            vio1 += np.clip((minn - y_pred), 0, None)/4
            vio2 += np.clip((y_pred - maxx), 0, None)/4

        text_red = 'TRV+:{:.1f}[°C-h]'.format(vio2[0])
        text_blue = 'TRV-:{:.1f}[°C-h]'.format(vio1[0])
        if self.args.modeltype == 'modnn':
            pad = 0.53
        else:
            pad = 0.49
        ax.text(pad, 0.15, text_red, fontsize=9, color='red', transform=ax.transAxes)
        ax.text(pad, 0.01, text_blue, fontsize=9, color='blue', transform=ax.transAxes)

        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=9, frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=9)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.xaxis.set_minor_locator(dates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b%d'))
        ax.set_xlabel(None)
        ax.set_ylabel('Temperature[°C]', fontsize=9)
        ax.margins(x=0)
        # ax.set_yticks(np.arange(8, 32, 5))
        # ax.set_ylim(10, 30)
        plt.close()
        folder = '../Saved/Overall'
        if not os.path.exists(folder):
            os.makedirs(folder)
        plot_name = 'Temp[{}].png'.format(self.dataset.test_raw_df.index[0])
        saveplot = os.path.join(folder, plot_name)
        fig.savefig(saveplot)

    def grad_check(self):
        Joc_matrix_list = grad_model(model=self.model, test_loader=self.dataset.TestLoader, device=self.device)
        Joc_matrix_ave = np.zeros_like(Joc_matrix_list[0])
        for l in range(len(Joc_matrix_list)):
            Joc_matrix_ave += Joc_matrix_list[l]
        Joc_matrix_ave = Joc_matrix_ave / len(Joc_matrix_list)

        # Define the check terms and corresponding labels
        check_terms = {
            "HVAC": {"index": 6, "ylabel": "HVAC Power (Timestep)", "xlabel": "Zone Temp (Timestep)"},
            "OA": {"index": 1, "ylabel": "Outdoor Air Temp (Timestep)", "xlabel": "Zone Temp (Timestep)"},
            "Solar": {"index": 2, "ylabel": "Solar Radiation (Timestep)", "xlabel": "Zone Temp (Timestep)"},
            "Occ": {"index": 5, "ylabel": "Occupancy (Timestep)", "xlabel": "Zone Temp (Timestep)"}
        }

        for name, meta in check_terms.items():
            grad_matrix = Joc_matrix_ave[:, :, [meta["index"]]].squeeze()

            fig, ax = plt.subplots(1, 1, figsize=(2.84, 1.92), dpi=300, constrained_layout=True)
            heatmap = sns.heatmap(
                grad_matrix[48:48 + 96, 48:48 + 96].T, ax=ax,
                cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True),
                cbar_kws={'label': None}
            )

            ax.set_ylabel(meta["ylabel"], fontsize=9)
            ax.set_xlabel(meta["xlabel"], fontsize=9)
            ax.tick_params(axis='both', which='both', labelsize=5)
            ax.margins(x=0)

            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label('Gradient', size=9)

            plt.title(f'Jacobian Heatmap: {name}', fontsize=10)

            plt.savefig(f'96 Steps Jacobian Heatmap: {name}.png', dpi=300)
            plt.show()



