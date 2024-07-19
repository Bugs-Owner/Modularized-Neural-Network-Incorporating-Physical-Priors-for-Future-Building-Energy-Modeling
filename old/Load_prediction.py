import time
import math
from DataPrepare import DataCook
import Net
import matplotlib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
matplotlib.use('TkAgg',force=True)
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class ddpred:
    def __init__(self):
        self.datapath = None
        self.dataset = None
        self.models = None
        self.para = None
        self.mdl = None
        self.wea = None
        self.delta = None
        self.city = None
        self.timespan = None

    def data_ready(self, num_zone, enLen, deLen, startday, trainday, testday,
                   resolution, training_batch, adj, wea, delta, city, timespan, mode):
        self.wea = wea
        self.delta = delta
        self.mode = mode
        self.city = city
        self.timespan = timespan
        print('Preparing data')
        if num_zone == 1:
            print("Single Zone Task")
            self.datapath = ("../EnergyPlus/EP_Training_Data/Single"+
                             '/Train_{}_{}_{}_{}_{}.csv'.format(delta, city, timespan, wea, mode))
            self.mdl = "Single"
        else:
            print("Multi Zone Task")
            self.datapath = ("../EnergyPlus/EP_Training_Data/Multi"+
                             '\Train_{}_{}_{}_{}_{}.csv'.format(delta, city, timespan, wea, mode))
            self.mdl = "Multi"
        start_time = time.time()
        DC = DataCook()
        DC.data_preprocess(datapath=self.datapath, num_zone=num_zone, adj_matrix=adj, mode=mode)
        DC.data_roll(enLen, deLen, startday, trainday, testday, resolution)
        DC.data_loader(training_batch)
        self.dataset = DC
        print("--- %s seconds ---" % (time.time() - start_time))

    def train(self, para):
        self.para = para
        print('Training start')
        start_time_sub = time.time()
        start_time_total = time.time()

        if para["type"]=="Single":
            models = {}
            for zone in range(self.dataset.num_zone):
                print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
                para["adjMatrix"] = torch.from_numpy(self.dataset.normalized_matrix.astype('float32'))
                model = Net.gru_seq2seq()
                model.initial(para)
                model = model.cuda()
                model.train_load_estimation_model(dataloder=self.dataset.TrainLoader[zone], zone_index=zone)
                models[zone] = model
                print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
                print("--- %s seconds ---" % (time.time() - start_time_sub))
                start_time_sub = time.time()
                folder_path = "./TrainedModel/" + para["Task"] + para["type"] + para["mode"]
                modelname = '[{} to {}]Model{}.pth'.format(self.dataset.train_start, self.dataset.train_end, zone)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                savemodel = os.path.join(folder_path, modelname)
                torch.save(model.state_dict(), savemodel)
            self.models = models
        else:
            models = {}
            for zone in range(self.dataset.num_zone):
                print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
                para["adjMatrix"] = torch.from_numpy(self.dataset.normalized_matrix.astype('float32'))
                model = Net.gru_seq2seq()
                model.initial(para)
                model = model.cuda()
                model.multi_train_load_estimation_model(dataloder=self.dataset.TrainLoader[zone], zone_index=zone)
                models[zone] = model
                print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
                print("--- %s seconds ---" % (time.time() - start_time_sub))
                start_time_sub = time.time()
                folder_path = "./TrainedModel/" + para["Task"] + para["type"] + para["mode"]
                #Changed here
                modelname = '[{} to {}]Model{}.pth'.format(self.dataset.train_start, self.dataset.train_end, zone)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                savemodel = os.path.join(folder_path, modelname)
                torch.save(model.state_dict(), savemodel)
            self.models = models

        print("--- %s seconds ---" % (time.time() - start_time_total))

    def load(self, para):
        para["adjMatrix"] = torch.from_numpy(self.dataset.normalized_matrix.astype('float32'))
        self.para = para
        print('Loading start')
        start_time = time.time()
        models = {}
        for zone in range(self.dataset.num_zone):
            print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
            model = Net.gru_seq2seq()
            model.initial(self.para)
            model = model.cuda()
            folder_path = "./TrainedModel/" + para["Task"] + para["type"] + para["mode"]
            modelname = '[{} to {}]Model{}.pth'.format(self.dataset.train_start, self.dataset.train_end, zone)
            loadmodel = os.path.join(folder_path, modelname)
            model.load_state_dict(torch.load(loadmodel))
            model.eval()
            models[zone] = model
            print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
        self.models = models
        print("--- %s seconds ---" % (time.time() - start_time))

    def transfer(self, para):
        para["adjMatrix"] = torch.from_numpy(self.dataset.normalized_matrix.astype('float32'))
        self.para = para
        print('Transfer start')
        start_time = time.time()
        models = {}
        for zone in range(self.dataset.num_zone):
            print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
            model = Net.gru_seq2seq()
            model.initial(para)
            model = model.cuda()
            folder_path = "./TrainedModel/" + para["Task"] + para["type"] + para["mode"]
            modelname = '[{} to {}]Model{}.pth'.format(self.dataset.train_start, self.dataset.train_end, zone)
            loadmodel = os.path.join(folder_path, modelname)
            model.load_state_dict(torch.load(loadmodel))
            # freeze everything
            for param in model.parameters():
                param.requires_grad = False
            for param in model.encoder_Ehvac.parameters():
                param.requires_grad = True
            for param in model.decoder_Ehvac.parameters():
                param.requires_grad = True
            models[zone] = model
            print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
        self.models = models
        print("--- %s seconds ---" % (time.time() - start_time))

    def test(self):
        print('Testing start')
        start_time = time.time()
        for zone in range(self.dataset.num_zone):
            self.models[zone].test_load_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                                         loadscal=self.dataset.processed_data[zone]['PhvacScaler'])
        print("--- %s seconds ---" % (time.time() - start_time))

    def multitest(self):
        print('Testing start')
        start_time = time.time()
        for zone in range(self.dataset.num_zone):
            self.models[zone].multi_test_load_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                                         loadscal=self.dataset.processed_data[zone]['PhvacScaler'])
        print("--- %s seconds ---" % (time.time() - start_time))

    def Retrain(self, feature):
        self.feature = feature
        print('Testing start')
        start_time = time.time()
        if feature == 'phvac':
            self.Scaler_ = 'PhvacScaler'
            y_index = 7
        if feature == 'HVAC':
            self.Scaler_ = 'EhvacScaler'
            y_index = -4
        if feature == 'Etotal':
            self.Scaler_ = 'TotalScaler'
            y_index = -3
        if feature == 'Ecool':
            self.Scaler_ = 'TotalScaler'
            y_index = -2
        if feature == 'Eheat':
            self.Scaler_ = 'TotalScaler'
            y_index = -1

        Emodels = {}
        for zone in range(self.dataset.num_zone):
            print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
            model = self.models[zone]
            model.train_energy_estimation_model(dataloder=self.dataset.TrainLoader[zone],zone_index=zone, y_index=y_index)
            Emodels[zone] = model
            print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
            folder_path = "./TrainedModel/" + self.para["Task"] + self.para["type"] + self.para["mode"]
            modelname = '[{} to {}]Model{}{}.pth'.format(self.dataset.test_start, self.dataset.test_end, zone, feature)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            savemodel = os.path.join(folder_path, modelname)
            torch.save(model.state_dict(), savemodel)
        self.models = Emodels

        for zone in range(self.dataset.num_zone):
            self.models[zone].test_energy_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                         loadscal=self.dataset.processed_data[zone][self.Scaler_])

        print("--- %s seconds ---" % (time.time() - start_time))

    def reload(self, para, feature):
        para["adjMatrix"] = torch.from_numpy(self.dataset.normalized_matrix.astype('float32'))
        self.para = para
        self.feature = feature
        if feature == 'phvac':
            self.Scaler_ = 'PhvacScaler'
        if feature == 'HVAC':
            self.Scaler_ = 'EhvacScaler'
        if feature == 'Etotal':
            self.Scaler_ = 'TotalScaler'
        if feature == 'Ecool':
            self.Scaler_ = 'EcoolScaler'
        if feature == 'Eheat':
            self.Scaler_ = 'EheatScaler'
        print('Loading start')
        start_time = time.time()
        models = {}
        for zone in range(self.dataset.num_zone):
            print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
            model = Net.gru_seq2seq()
            model.initial(self.para)
            model = model.cuda()
            folder_path = "./TrainedModel/" + self.para["Task"] + self.para["type"] + self.para["mode"]
            modelname = '[{} to {}]Model{}{}.pth'.format(self.dataset.test_start, self.dataset.test_end, zone, self.feature)
            loadmodel = os.path.join(folder_path, modelname)
            model.load_state_dict(torch.load(loadmodel))
            model.eval()
            models[zone] = model
            print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
        self.models = models
        for zone in range(self.dataset.num_zone):
            self.models[zone].test_energy_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                         loadscal=self.dataset.processed_data[zone][self.Scaler_])
        print("--- %s seconds ---" % (time.time() - start_time))

    def multiplot(self, feature):
        print('Ploting start')
        start_time = time.time()
        if feature == 'Cphvac':
            ylabel = 'HVAC Load[kW]'
        if feature == 'HVAC':
            ylabel = 'HVAC Energy[kWh]'
        if feature == 'Ecool':
            ylabel = 'HVAC Energy[kWh]'
        if feature == 'Eheat':
            ylabel = 'HVAC Energy[kWh]'
        if feature == 'Etotal':
            ylabel = 'Building Total Load[W]'
        rawdf = self.dataset.test_raw_df
        fig, axes = plt.subplots(5, 1, figsize=(15, 9), dpi=300, sharey='row',
                                 gridspec_kw={'hspace': 0.1}, constrained_layout=True)
        test_len = len(self.models[0].de_denorm)
        pred_len = self.dataset.deLen

        for row in range(self.dataset.num_zone):
            meas = rawdf['{}_{}'.format(feature, row)].values[:test_len] / 1000
            pred = np.array(self.models[row].de_denorm)[:, [0], :].squeeze() / 1000
            errdf = pd.DataFrame({'Groundtruth': meas, 'Prediction': pred})
            start_date = '2023-01-01 00:00:00'  # This is an arbitrary start date to generate index
            time_index = pd.date_range(start=start_date, periods=len(errdf), freq='15T')
            errdf.index = time_index
            errdf = errdf.resample('1H').mean()
            mae = mean_absolute_error(errdf["Groundtruth"], errdf["Prediction"])
            r2 = r2_score(errdf["Groundtruth"], errdf["Prediction"])
            mbe = (errdf["Prediction"] - errdf["Groundtruth"]).mean() * 100
            rmse = np.sqrt(((abs(errdf["Prediction"]) - abs(errdf["Groundtruth"])) ** 2).mean())
            mean_observed = abs(errdf['Groundtruth']).mean()
            cv_rmse = rmse / mean_observed * 100

            axes[row].plot_date(rawdf.index[:test_len], rawdf['{}_{}'.format(feature, row)].values[:test_len]/1000, '-',
                         linewidth=1.5, color="#159A9C", label='Measurement')
            for timestep in range(test_len):
                axes[row].plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                             (self.models[row].de_denorm[timestep][:test_len - timestep]/1000), '--', linewidth=0.4,
                             color="#EA7E7E")
            axes[row].plot_date(rawdf.index[0:0 + 1],
                         (self.models[row].de_denorm[0][0]/1000), '--', linewidth=1.5, color="#EA7E7E",
                         label='SeqPINN Prediction')
            # axes[row].text(0.01, 0.05, f'R2: {r2:.2f}  MBE: {mbe:.2f}[%]  MAE: {mae:.2f}[kW]  CV(RMSE): {cv_rmse:.2f}[%]', size=16,
            #                 transform=axes[row].transAxes)
            axes[row].legend([], [], frameon=False)
            axes[row].set_xticklabels([])
            axes[row].set_xlabel(None)
            axes[row].set_ylabel(None)
            axes[row].margins(x=0)
            axes[row].tick_params(axis='both', which='minor', labelsize=12)
            axes[row].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_ylim(-8, 0.2)
        axes[0].set_yticks(np.arange(-8, 0.1, 2))
        axes[1].set_ylim(-8, 0.2)
        axes[1].set_yticks(np.arange(-8, 0.1, 2))
        axes[2].set_ylim(-8, 0.2)
        axes[2].set_yticks(np.arange(-8, 0.1, 2))
        axes[3].set_ylim(-8, 0.2)
        axes[3].set_yticks(np.arange(-8, 0.1, 2))
        axes[4].set_ylim(-8, 0.2)
        axes[4].set_yticks(np.arange(-8, 0.1, 2))
        axes[row].tick_params(axis='both', which='minor', labelsize=12)
        axes[row].tick_params(axis='both', which='major', labelsize=12)
        axes[row].xaxis.set_minor_locator(dates.DayLocator(interval=1))
        axes[row].xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        axes[row].xaxis.set_major_locator(dates.MonthLocator(interval=1))
        axes[row].xaxis.set_major_formatter(dates.DateFormatter('%b'))
        axes[0].legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16, frameon=False)
        fig.supylabel(ylabel, fontsize=16, fontweight='bold')
        plt.close(fig)
        folder_path = "load_plot"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = '{}_{}_{}_{}_{}_{}_Multi_Temp[{} to {}].png'.format(feature, self.mode, self.delta, self.city, self.timespan, self.wea,
                                                                  self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot)


        # csvname = '{}_{}_{}_{}_{}_{}_Multi_Temp[{} to {}].csv'.format(feature, self.mode, self.delta, self.city, self.timespan, self.wea,
        #                                                           self.dataset.test_start, self.dataset.test_end)
        # savecsv = os.path.join(folder_path, csvname)
        # errdf.to_csv(savecsv)
        print("--- %s seconds ---" % (time.time() - start_time))
    def multi_csv_save(self, feature):
        print('Ploting start')
        start_time = time.time()
        if feature == 'phvac':
            if self.mode == "Cooling":
                self.feature = "Cphvac"
            else:
                self.feature = "Hphvac"
        else:
            self.feature = feature
        rawdf = self.dataset.test_raw_df
        test_len = len(self.models[0].de_denorm)
        for zone in range(self.dataset.num_zone):
            meas = rawdf['{}_{}'.format(self.feature, zone)].values[:test_len] / 1000
            pred = np.array(self.models[zone].de_denorm)[:, [0], :].squeeze() / 1000
            errdf = pd.DataFrame({'Groundtruth': meas, 'Prediction': pred})
            start_date = '2023-01-01 00:00:00'  # This is an arbitrary start date
            time_index = pd.date_range(start=start_date, periods=len(errdf), freq='15T')
            errdf.index = time_index
            errdf = errdf.resample('1H').mean()
            #errdf = errdf.resample('1D').mean()
            folder_path = "load_plot"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            csvname = '{}_{}_{}_{}_{}_{}_Multi_{}[{} to {}].csv'.format(self.feature, self.mode, self.delta, self.city, self.timespan, self.wea,
                                                                      zone, self.dataset.test_start, self.dataset.test_end)
            savecsv = os.path.join(folder_path, csvname)
            errdf.to_csv(savecsv)
        print("--- %s seconds ---" % (time.time() - start_time))
    def csv_save(self, feature):
        print('Ploting start')
        start_time = time.time()
        if feature == 'phvac':
            if self.mode == "Cooling":
                self.feature = "Cphvac"
            else:
                self.feature = "Hphvac"
        else:
            self.feature = feature
        rawdf = self.dataset.test_raw_df
        test_len = len(self.models[0].de_denorm)
        meas = rawdf['{}_{}'.format(self.feature, 0)].values[:test_len] / 1000
        pred = np.array(self.models[0].de_denorm)[:, [0], :].squeeze() / 1000
        errdf = pd.DataFrame({'Groundtruth': meas, 'Prediction': pred})
        start_date = '2023-01-01 00:00:00'  # This is an arbitrary start date
        time_index = pd.date_range(start=start_date, periods=len(errdf), freq='15T')
        errdf.index = time_index
        errdf = errdf.resample('1H').mean()
        #errdf = errdf.resample('1D').mean()
        folder_path = "load_plot"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        csvname = '{}_{}_{}_{}_{}_{}_Single_Temp[{} to {}].csv'.format(self.feature, self.mode, self.delta, self.city, self.timespan, self.wea,
                                                                  self.dataset.test_start, self.dataset.test_end)
        savecsv = os.path.join(folder_path, csvname)
        errdf.to_csv(savecsv)
        print("--- %s seconds ---" % (time.time() - start_time))

    def plot(self, feature):
        print('Ploting start')
        start_time = time.time()
        if feature == 'phvac':
            if self.mode == "Cooling":
                self.feature = "Cphvac"
            else:
                self.feature = "Hphvac"
            ylabel = 'HVAC Load[kW]'
        if feature == 'HVAC':
            ylabel = 'HVAC Energy[kWh]'
        if feature == 'Ecool':
            ylabel = 'HVAC Energy[kWh]'
        if feature == 'Eheat':
            ylabel = 'HVAC Energy[kWh]'
        elif feature == 'Etotal':
            ylabel = 'Total Load[kW]'
        rawdf = self.dataset.test_raw_df
        fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=300, constrained_layout=True)
        test_len = len(self.models[0].de_denorm)
        pred_len = self.dataset.deLen
        meas = rawdf['{}_{}'.format(self.feature, 0)].values[:test_len] / 1000
        pred = np.array(self.models[0].de_denorm)[:, [0], :].squeeze() / 1000
        # plt.plot(meas)
        # plt.plot(pred)
        # plt.show()
        # pred = np.clip(pred, 0, pred.max())
        errdf = pd.DataFrame({'Groundtruth': meas, 'Prediction': pred})
        start_date = '2023-01-01 00:00:00'  # This is an arbitrary start date
        time_index = pd.date_range(start=start_date, periods=len(errdf), freq='15T')
        errdf.index = time_index
        errdf = errdf.resample('1H').mean()
        mae = mean_absolute_error(errdf["Groundtruth"], errdf["Prediction"])
        r2 = r2_score(errdf["Groundtruth"], errdf["Prediction"])
        mbe = (errdf["Prediction"] - errdf["Groundtruth"]).mean() * 100
        nmbe = mbe/(errdf['Groundtruth'].mean())
        rmse = np.sqrt(np.mean((errdf["Prediction"] - errdf["Groundtruth"]) ** 2))
        mean_actual = np.mean(errdf["Groundtruth"])
        cv_rmse = (rmse / np.abs(mean_actual)) * 100

        ax.plot_date(rawdf.index[:test_len], rawdf['{}_{}'.format(self.feature, 0)].values[:test_len]/1000, '-',
                     linewidth=1.5, color="#159A9C", label='Measurement')
        for timestep in range(test_len):
            y = (self.models[0].de_denorm[timestep][:test_len - timestep])/1000
            # y = np.clip(y, 0, y.max())
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         y, '--', linewidth=0.4,
                         color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1],
                     (self.models[0].de_denorm[0][0])/1000, '--', linewidth=1.5, color="#EA7E7E",
                     label='SeqPINN Prediction')
        # ax.text(0.01, 0.07, f'R2: {r2:.2f}  MBE: {mbe:.2f}[%]  MAE: {mae:.2f}[kW]', size=16,
        #                 transform=ax.transAxes)

        # ax.text(0.01, 0.85, f'R2: {r2:.2f}  MBE: {mbe:.2f}[%]  MAE: {mae:.2f}[kW]', size=16,
        #                 transform=ax.transAxes)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16, frameon=False)

        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
        ax.margins(x=0)
        plt.show()
        # plt.close(fig)
        folder_path = "load_plot"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = '{}_{}_{}_{}_{}_{}_Single_Temp[{} to {}].png'.format(self.feature, self.mode, self.delta, self.city, self.timespan, self.wea,
                                                                  self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot)
        plotname = '{}_{}_{}_{}_{}_{}_Single_Temp[{} to {}].pdf'.format(self.feature, self.mode, self.delta, self.city, self.timespan, self.wea,
                                                                  self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot, transparent=True, format='pdf')

        # csvname = '{}_{}_{}_{}_{}_{}_Single_Temp[{} to {}].csv'.format(self.feature, self.mode, self.delta, self.city, self.timespan, self.wea,
        #                                                           self.dataset.test_start, self.dataset.test_end)
        # savecsv = os.path.join(folder_path, csvname)
        # errdf.to_csv(savecsv)
        print("--- %s seconds ---" % (time.time() - start_time))

    
