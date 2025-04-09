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
from modnn.Models import ModNN, BaseNN
from modnn.Play import train_model, test_model, check_model, dynamic_check_model
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
            model = ModNN.ModNN(self.args).to(self.device)
        if self.args["modeltype"] == "LSTM":
            model = BaseNN.Baseline(self.args).to(self.device)
        start_time = time.time()
        print('model training')
        self.model, self.loss_dic = train_model(model=model,
                                                train_loader=self.dataset.TrainLoader,
                                                valid_loader=self.dataset.ValidLoader,
                                                test_loader =self.dataset.TestLoader,
                                                lr=self.args["para"]['lr'],
                                                epochs=self.args["para"]['epochs'],
                                                patience=self.args["para"]['patience'],
                                                tempscal = self.dataset.scalers['temp_room'],
                                                fluxscal = self.dataset.scalers['phvac'],
                                                enlen = self.args['enLen'],
                                                delen=self.args['deLen'],
                                                rawdf = self.dataset.test_raw_df,
                                                plott = self.args["plott"],
                                                modeltype = self.args["modeltype"],
                                                scale = self.args["scale"],
                                                device=self.device)
        print("--- %s seconds ---" % (time.time() - start_time))

        folder_name = "../Saved/Trained_mdl" + 'Enco{}_Deco{}'.format(str(self.args['enLen']), str(self.args['deLen']))
        mdl_name = '{}_{}daysTest_on{}.pth'.format(self.args["modeltype"], str(self.args["trainday"]), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savemodel = os.path.join(folder_name, mdl_name)
        torch.save(model.state_dict(), savemodel)

        folder_name = "../Saved/Loss" + 'Enco{}_Deco{}'.format(str(self.args['enLen']), str(self.args['deLen']))
        loss_name = '{}Loss{}days_Test_on{}.pickle'.format(self.args["modeltype"], str(self.args["trainday"]), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        saveloss = os.path.join(folder_name, loss_name)
        with open(saveloss, 'wb') as f:
            pickle.dump(self.loss_dic, f)

        # #Loss display

        # def _loss_norm(loss):
        #     loss_arr = np.array(loss)
        #     return np.log(loss_arr)
        #
        # color_paletteZ_ = ["#008744", "#0057e7", "#d62d20", "#ffa700"]
        # fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=300, constrained_layout=True)
        # ax.plot(_loss_norm(self.loss_dic['train_temp_losses']), label='Temp_mse', color='#f24e4c')
        # ax.plot(_loss_norm(self.loss_dic['train_diff_losses']), label='Temp_diff', color='#bbe088')
        #
        # # ax.plot(self.loss_dic['valid_losses'], label='Validation_loss', color='#bbe088')
        #
        # ax.plot(_loss_norm(self.loss_dic['vio_positive_loss']), label='TRV+', color='#537af5')
        # ax.plot(_loss_norm(self.loss_dic['vio_negative_loss']), label='TRV-', color='#f26fc4')
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
        ax.plot(self.loss_dic['train_losses'], label='Training', color='#F0907F')
        ax.plot(self.loss_dic['valid_losses'], label='Validation', color='#8B90F5')
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
        axins.plot(self.loss_dic['train_losses'], label='Training_loss', color='#F0907F')
        axins.plot(self.loss_dic['valid_losses'], label='Validation loss', color='#8B90F5')

        # Specify the limits of your zoom-in area
        # x1, x2, y1, y2 = 50, 120, 0.001, 0.008  # These limits should be set according to your data
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)

        # Optionally add grid, customize ticks, etc.
        axins.grid(True)
        axins.tick_params(axis='both', which='both', labelsize=5)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.13), ncol=1, fontsize=7, frameon=False)
        plt.show()

    def load(self, mdl_name=None):
        start_time = time.time()
        print("Loading model")
        if "modnn" in self.args["modeltype"]:
            model = ModNN.ModNN(self.args).to(self.device)
        if self.args["modeltype"] == "LSTM":
            model = BaseNN.Baseline(self.args).to(self.device)

        folder_name = "../Saved/Trained_mdl" + 'Enco{}_Deco{}'.format(str(self.args['enLen']), str(self.args['deLen']))
        if mdl_name is None:
            mdl_name = '{}_{}daysTest_on{}.pth'.format(self.args["modeltype"], str(self.args["trainday"]), self.dataset.test_start)
        else:
            mdl_name=mdl_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        loadmodel = os.path.join(folder_name, mdl_name)
        model.load_state_dict(torch.load(loadmodel))
        model.eval()
        self.model = model
        print("--- %s seconds ---" % (time.time() - start_time))

    def test(self):
        print('Testing start')
        start_time = time.time()
        def MAPE(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        def MAE(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs(y_true - y_pred))

        self.to_out, self.en_out, self.de_out  = test_model(model=self.model, test_loader=self.dataset.TestLoader, device=self.device,
                             tempscal=self.dataset.scalers['temp_room'], enlen=self.args['enLen'])

        folder_name = "../Result/{}/".format(str(self.args["modeltype"])) + 'Enco{}_Deco{}'.format(str(self.args['enLen']), str(self.args['deLen']))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_name = '(raw)Train_with_{}days\nTest_on{}.csv'.format(str(self.args["trainday"]), self.dataset.test_start)
        savecsv = os.path.join(folder_name, csv_name)
        self.dataset.test_raw_df.to_csv(savecsv)
        test_len = len(self.de_out)
        pred_len = self.args['deLen']
        test_result = {}
        mae = {}
        mape = {}
        gdtruth = (self.dataset.test_raw_df['temp_room'].values - 32) * 5 / 9
        for timestep in range(test_len):
            tem = self.de_out[timestep]
            tem = (tem - 32) * 5 / 9
            test_result[timestep] = tem
            y_true = gdtruth[timestep: timestep + pred_len].reshape(-1, 1)
            y_mear = self.de_out[timestep]
            y_mear = (y_mear - 32) * 5 / 9
            mae[timestep] = MAE(y_true=y_true, y_pred=y_mear)
            mape[timestep] = MAPE(y_true=y_true, y_pred=y_mear)
        self.mae = mae
        self.mape = mape
        dic_name = '(pred)Train_with_{}days\nTest_on{}.pkl'.format(str(self.args["trainday"]), self.dataset.test_start)
        savedic = os.path.join(folder_name, dic_name)
        with open(savedic, 'wb') as f:
            pickle.dump(test_result, f)
        mae_name = '(mae)Train_with_{}days\nTest_on{}.pkl'.format(str(self.args["trainday"]), self.dataset.test_start)
        savemae = os.path.join(folder_name, mae_name)
        with open(savemae, 'wb') as f:
            pickle.dump(mae, f)
        mape_name = '(mape)Train_with_{}days\nTest_on{}.pkl'.format(str(self.args["trainday"]), self.dataset.test_start)
        savemape = os.path.join(folder_name, mape_name)
        with open(savemape, 'wb') as f:
            pickle.dump(mape, f)
        print("--- %s seconds ---" % (time.time() - start_time))

    def check(self):
        print('Checking start')
        start_time = time.time()

        (self.outputs_max_denorm, self.outputs_min_denorm, self.outputs_denorm,
         self.cooling_max_denorm, self.cooling_min_denorm) = check_model(model=self.model,
                                                                        test_loader=self.dataset.TestLoader,
                                                                        tempscal=self.dataset.scalers['temp_room'],
                                                                        hvacscale=self.dataset.scalers['phvac'],
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
                                                                    tempscal=self.dataset.scalers['temp_room'],
                                                                    hvacscale=self.dataset.scalers['phvac'],
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
        return vio, self.mae, self.loss_dic

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

    def grad_check(self, check_term="HVAC"):

        Joc_input = torch.rand(1, 144, 7).to(self.device)
        self.model.train()

        def extract(Joc_input):
            output, _, _ = self.model(Joc_input)
            return output

        Joc_matrix = F.jacobian(extract, Joc_input).detach().cpu().numpy().squeeze()

        # Dictionary mapping term to corresponding index and label
        term_map = {
            "HVAC": {"index": 6, "ylabel": "HVAC Input (Timestep)"},
            "SpaceT": {"index": 0, "ylabel": "Space Temp (Timestep)"},
            "OA": {"index": 1, "ylabel": "Outdoor Air (Timestep)"},
            "Solar": {"index": 2, "ylabel": "Solar Radiation (Timestep)"}
        }

        if check_term not in term_map:
            raise ValueError(f"Invalid check_term '{check_term}'. Choose from: {list(term_map.keys())}")

        idx = term_map[check_term]["index"]
        ylabel = term_map[check_term]["ylabel"]
        Joc_selected = Joc_matrix[:, :, [idx]].squeeze()

        fig, ax = plt.subplots(1, 1, figsize=(2.84, 1.92), dpi=300, constrained_layout=True)
        heatmap = sns.heatmap(Joc_selected[48:, 48:].T, ax=ax, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True),
                              cbar_kws={'label': None})

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('Temperature Output (Timestep)', fontsize=9)
        ax.tick_params(axis='both', which='both', labelsize=5)
        ax.margins(x=0)

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label('Gradient', size=9)

        plt.show()



