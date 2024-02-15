from Dataset import DataCook
import pickle
import os
import time
import Net
import Net_noencoder
import matplotlib
import numpy as np
import torch
import matplotlib.dates as dates
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class ddpred:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.para = None
    def data_ready(self, path, enLen, deLen, startday, trainday, testday, resolution, training_batch):
        print('Preparing data')
        start_time = time.time()
        DC = DataCook()
        DC.data_preprocess(datapath=path, num_zone=1)
        DC.data_roll(enLen, deLen, startday, trainday, testday, resolution)
        DC.data_loader(training_batch)
        self.dataset = DC
        print("--- %s seconds ---" % (time.time() - start_time))
    def train(self, para):
        self.para = para
        print('Training start')
        start_time = time.time()
        if para["encoLen"] != 0:
            model = Net.gru_seq2seq(para)
        else:
            model = Net_noencoder.gru_seq2seq(para)
        model = model.cuda()
        model.train_model(dataloder=self.dataset.TrainLoader[0])
        print("--- %s seconds ---" % (time.time() - start_time))
        self.model = model
        print('Saving start')
        start_time = time.time()
        folder_name = "JouleModel/" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        mdl_name = 'Train_with_{}days\nTest_on{}.pth'.format(str(self.dataset.trainday), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savemodel = os.path.join(folder_name, mdl_name)
        torch.save(model.state_dict(), savemodel)
        print("--- %s seconds ---" % (time.time() - start_time))
    def load(self, para):
        print('Loading start')
        start_time = time.time()
        self.para = para
        model = Net.gru_seq2seq(para)
        model = model.cuda()
        folder_name = "JouleModel/" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        mdl_name = 'Train_with_{}days\nTest_on{}.pth'.format(str(self.dataset.trainday), self.dataset.test_start)
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
        self.model.test_model(dataloder=self.dataset.TestLoader[0],
                              tempscal=self.dataset.processed_data[0]['TzoneScaler'])
        print("--- %s seconds ---" % (time.time() - start_time))
        folder_name = "JouleModelResult/" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_name ='(raw)Train_with_{}days\nTest_on{}.csv'.format(str(self.dataset.trainday), self.dataset.test_start)
        savecsv = os.path.join(folder_name, csv_name)
        self.dataset.test_raw_df.to_csv(savecsv)
        test_len = len(self.model.de_denorm)
        pred_len = self.dataset.deLen
        test_result = {}
        mae = {}
        mape = {}
        gdtruth = (self.dataset.test_raw_df['temp_zone_{}'.format(0)].values - 32) * 5 / 9
        for timestep in range(test_len):
            tem = self.model.de_denorm[timestep]
            tem = (tem - 32) * 5 / 9
            test_result[timestep] = tem
            y_true = gdtruth[timestep: timestep + pred_len].reshape(-1,1)
            y_mear = self.model.de_denorm[timestep]
            y_mear = (y_mear - 32) * 5 / 9
            mae[timestep] = MAE(y_true=y_true, y_pred=y_mear)
            mape[timestep] = MAPE(y_true=y_true, y_pred=y_mear)
        self.mae=mae
        self.mape=mape
        dic_name = '(pred)Train_with_{}days\nTest_on{}.pkl'.format(str(self.dataset.trainday), self.dataset.test_start)
        savedic = os.path.join(folder_name, dic_name)
        with open(savedic, 'wb') as f:
            pickle.dump(test_result, f)
        mae_name = '(mae)Train_with_{}days\nTest_on{}.pkl'.format(str(self.dataset.trainday), self.dataset.test_start)
        savemae = os.path.join(folder_name, mae_name)
        with open(savemae, 'wb') as f:
            pickle.dump(mae, f)
        mape_name = '(mape)Train_with_{}days\nTest_on{}.pkl'.format(str(self.dataset.trainday), self.dataset.test_start)
        savemape = os.path.join(folder_name, mape_name)
        with open(savemape, 'wb') as f:
            pickle.dump(mape, f)
    def prediction_show(self):
        print('Ploting start')
        rawdf = self.dataset.test_raw_df
        test_len = len(self.model.de_denorm)
        pred_len = self.dataset.deLen
        fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=300, sharex='col', sharey='row', constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len]-32)*5/9, '-',linewidth=1.5, color="#159A9C", label='Measurement')
        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        for timestep in range(test_len):
            tem=self.model.de_denorm[timestep][:test_len-timestep]
            tem=(tem-32)*5/9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         tem, '--', linewidth=0.4, color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1],((self.model.de_denorm[0][0])-32)*5/9, '--', linewidth=1.5, color="#EA7E7E",label='SeqPINN Prediction')
        ax.text(0.01, 0.85, 'MAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(mae,mape), fontsize=10, color='gray', fontweight='bold',transform=ax.transAxes)
        ax.legend([],[], frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
        ax.set_xlabel(None)
        ax.set_ylabel('Temperature[°C]', fontsize=16, fontweight='bold')
        ax.margins(x=0)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16, frameon=False)
        plt.show()

        folder_name = "JouleFigure/" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plot_name ='Train_with_{}days\nTest_on{}.png'.format(str(self.dataset.trainday), self.dataset.test_start)
        saveplot = os.path.join(folder_name, plot_name)
        fig.savefig(saveplot)


    def check(self):
        print('Checking start')
        start_time = time.time()
        checkdic = {}
        self.model.test_model(dataloder=self.dataset.TestLoader[0],
                               tempscal=self.dataset.processed_data[0]['TzoneScaler'])
        checkdic['Raw']=self.model.de_denorm
        for ch in ["HVAC", "Temp", "Sol"]:
            self.model.check_model(dataloder=self.dataset.TestLoader[0],
                                   tempscal=self.dataset.processed_data[0]['TzoneScaler'],
                                   check='{}min'.format(ch))
            checkdic['{}min'.format(ch)] = self.model.de_denorm
            self.model.check_model(dataloder=self.dataset.TestLoader[0],
                                   tempscal=self.dataset.processed_data[0]['TzoneScaler'],
                                   check='{}max'.format(ch))
            checkdic['{}max'.format(ch)] = self.model.de_denorm
        self.checkdic=checkdic
        print("--- %s seconds ---" % (time.time() - start_time))
        

    def check_show(self, check):
        print('Ploting start')
        rawdf = self.dataset.test_raw_df
        test_len = len(self.checkdic['Raw'])
        pred_len = self.dataset.deLen
        fig, ax = plt.subplots(1, 1, figsize=(3.1, 1.5), dpi=300, sharex='col', sharey='row', constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len]-32)*5/9, '-',linewidth=1, color="#159A9C", label='Measurement')
        for timestep in range(test_len):
            tem=self.checkdic['Raw'][timestep][:test_len-timestep]
            tem=(tem-32)*5/9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         tem, '--', linewidth=0.4, color="gray")
            minn=self.checkdic['{}min'.format(check)][timestep][:test_len-timestep]
            minn=(minn-32)*5/9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         minn, '--', linewidth=0.4, color="#8163FD")
            maxx=self.checkdic['{}max'.format(check)][timestep][:test_len-timestep]
            maxx=(maxx-32)*5/9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         maxx, '--', linewidth=0.4, color="#FF5F5D")

        ax.plot_date(rawdf.index[0:0 + 1],((self.checkdic['Raw'][0][0])-32)*5/9, '--', linewidth=1, color="gray",label='SeqPINN Prediction')
        ax.plot_date(rawdf.index[0:0 + 1], ((self.checkdic['{}min'.format(check)][0][0]) - 32) * 5 / 9, '--',
                     linewidth=1, color="#8163FD", label='Min {} sanity check'.format(check))
        ax.plot_date(rawdf.index[0:0 + 1], ((self.checkdic['{}max'.format(check)][0][0]) - 32) * 5 / 9, '--',
                     linewidth=1, color="#FF5F5D", label='Max {} sanity check'.format(check))
        ax.legend([],[], frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_minor_locator(dates.HourLocator(interval=2))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b%d'))
        ax.set_xlabel(None)
        ax.set_ylabel('Temperature[°C]', fontsize=7)
        ax.margins(x=0)
        #ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=7, frameon=False)
        ax.legend(loc='upper left', fontsize=7, frameon=False)
        plt.show()
        folder_path = "Joule_check"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = '{}.pdf'.format(check)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot, transparent=True, format='pdf')
