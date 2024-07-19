import time
from DataPrepare import DataCook
import Net
import matplotlib
from sklearn.metrics import mean_absolute_percentage_error, r2_score
matplotlib.use('TkAgg',force=True)
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import os

class ddpred:
    def __init__(self):
        self.datapath = None
        self.dataset = None
        self.model = None
        self.para = None
        self.mdl = None

        self.wea = None
        self.delta = None
        self.city = None
        self.timespan = None

    def data_ready(self, num_zone, enLen, deLen, startday, trainday, testday,
                   resolution, training_batch, tar, adj, wea, delta, city, timespan):
        self.wea = wea
        self.delta = delta
        self.city = city
        self.timespan = timespan
        print('Preparing data')
        if num_zone == 1:
            print("Single Zone Task")
            self.datapath = ("..\EnergyPlus\EP_Training_Data\Single"+
                             '\Train_{}_{}_{}_{}.csv'.format(delta, city, timespan, wea))
            self.mdl = "Single"
        else:
            print("Multi Zone Task")
            self.datapath = "..\EnergyPlus\EP_Training_Data\Multi\EP-train.csv"
            self.mdl = "Multi"
        start_time = time.time()
        DC = DataCook()
        DC.data_preprocess(datapath=self.datapath, num_zone=num_zone, adj_matrix=adj)
        DC.data_roll(enLen, deLen, startday, trainday, testday, resolution)
        DC.data_loader(training_batch, tar)
        self.dataset = DC
        print("--- %s seconds ---" % (time.time() - start_time))

    def train(self, para, action, load_start, load_end):
        print('Training start')
        self.para = para
        start_time_sub = time.time()
        start_time_total = time.time()

        if para["Task"] == "pHVAC_estimation":
            if action == 'train':
                models={}
                for zone in range(self.dataset.num_zone):
                    print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
                    model = Net.gru_seq2seq()
                    model.initial(para)
                    model = model.cuda()
                    model.train_load_estimation_model(dataloder=self.dataset.TrainLoader[zone], zone_index=zone)
                    models[zone] = model
                    print("{}/{} Finished".format(zone+1, self.dataset.num_zone))
                    print("--- %s seconds ---" % (time.time() - start_time_sub))
                    start_time_sub = time.time()
                    folder_path = "./TrainedModel/"+para["Task"]+para["type"]
                    modelname = '[{} to {}]Model{}.pth'.format(self.dataset.test_start, self.dataset.test_end, zone)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    savemodel = os.path.join(folder_path, modelname)
                    torch.save(model.state_dict(), savemodel)
                self.models = models
                print("--- %s seconds ---" % (time.time() - start_time_total))
            elif action == 'load':
                models = {}
                for zone in range(self.dataset.num_zone):
                    print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
                    model = Net.gru_seq2seq()
                    model.initial(para)
                    model = model.cuda()
                    folder_path = "./TrainedModel/" + para["Task"] + para["type"]
                    modelname = '[{} to {}]Model{}.pth'.format(load_start, load_end, zone)
                    loadmodel = os.path.join(folder_path, modelname)
                    model.load_state_dict(torch.load(loadmodel))
                    model.eval()
                    models[zone] = model
                    print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
                    print("--- %s seconds ---" % (time.time() - start_time_sub))
                    start_time_sub = time.time()
                self.models = models
                print("--- %s seconds ---" % (time.time() - start_time_total))
            elif action == 'energy':
                models = {}
                for zone in range(self.dataset.num_zone):
                    print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
                    model = Net.gru_seq2seq()
                    model.initial(para)
                    model = model.cuda()
                    folder_path = "./TrainedModel/" + para["Task"] + para["type"]
                    modelname = '[{} to {}]Model{}.pth'.format(load_start, load_end, zone)
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
                    print("--- %s seconds ---" % (time.time() - start_time_sub))
                    start_time_sub = time.time()
                self.models = models
                print("--- %s seconds ---" % (time.time() - start_time_total))

    def ReEnergy(self):
        para["Task"] == "pHVAC_estimation"
        print('Testing start')
        start_time_total = time.time()
        start_time_sub = time.time()
        Emodels = {}
        for zone in range(self.dataset.num_zone):
            print("{}/{} Started".format(zone + 1, self.dataset.num_zone))
            model = self.models[zone]
            model.train_energy_estimation_model(dataloder=self.dataset.TestLoader[zone],zone_index=zone)
            Emodels[zone] = model
            print("{}/{} Finished".format(zone + 1, self.dataset.num_zone))
            print("--- %s seconds ---" % (time.time() - start_time_sub))
            start_time_sub = time.time()
            folder_path = "./TrainedModel/" + self.para["Task"] + self.para["type"]
            modelname = '[{} to {}]Model{}.pth'.format(self.dataset.test_start, self.dataset.test_end, zone)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            savemodel = os.path.join(folder_path, modelname)
            torch.save(model.state_dict(), savemodel)
        self.Emodels = Emodels

        for zone in range(self.dataset.num_zone):
            self.models[zone].test_energy_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                         loadscal=self.dataset.processed_data[zone]['EhvacScaler'])

        print("--- %s seconds ---" % (time.time() - start_time_total))



    def test(self):
        print('Testing start')
        start_time = time.time()
        for zone in range(self.dataset.num_zone):
            self.models[zone].test_load_estimation_model(dataloder=self.dataset.TestLoader[zone],
                                         loadscal=self.dataset.processed_data[zone]['PhvacScaler'])
        print("--- %s seconds ---" % (time.time() - start_time))


    def plot(self, feature):
        print('Ploting start')
        start_time = time.time()
        rawdf = self.dataset.test_raw_df
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300, constrained_layout=True)
        test_len = len(self.models[0].de_denorm)
        pred_len = self.dataset.deLen

        ax.plot_date(rawdf.index[:test_len], rawdf['{}_{}'.format(feature,0)].values[:test_len], '-',
                     linewidth=1.5, color="#159A9C", label='Measurement')
        for timestep in range(test_len):
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         (self.models[0].de_denorm[timestep][:test_len-timestep]), '--', linewidth=0.4, color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1],
                     (self.models[0].de_denorm[0][0]), '--', linewidth=1.5, color="#EA7E7E",
                     label='SeqPINN Prediction')
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16, frameon=False)

        ax.set_ylabel('pHVAC[W]', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.tick_params(axis='both', which='major', labelsize=11)
        # ax.xaxis.set_minor_locator(dates.HourLocator(interval=2))
        # ax.xaxis.set_minor_formatter(dates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
        ax.margins(x=0)
        plt.close(fig)
        folder_path = "load_plot"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = '{}_{}_{}_{}_Single_Temp[{} to {}].png'.format(self.delta, self.city, self.timespan, self.wea,
                                                                  self.dataset.test_start, self.dataset.test_end)

        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot)
        print("--- %s seconds ---" % (time.time() - start_time))


    def testsave(self):
        print('Saving start')
        start_time = time.time()
        folder_path = "test_result"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = 'Test_{}_{}_{}_{}.pickle'.format(self.delta, self.city, self.timespan, self.wea)
        savefile = os.path.join(folder_path, filename)
        with open(savefile, 'wb') as handle:
            pickle.dump(self.models[0].de_denorm, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("--- %s seconds ---" % (time.time() - start_time))

    def loadsave(self):
        print('Loading start')
        start_time = time.time()
        folder_path = "test_result"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = 'Test_{}_{}_{}_{}.pickle'.format(self.delta, self.city, self.timespan, self.wea)
        savefile = os.path.join(folder_path, filename)
        with open(savefile, 'rb') as handle:
            pickle.load(handle)

        print("--- %s seconds ---" % (time.time() - start_time))
