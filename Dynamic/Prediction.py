import time
from DataPrepare import DataCook
import OnlySingel
import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from sklearn.metrics import mean_absolute_error, mean_squared_error
class ddpred:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.para = None

    def data_ready(self, path, enLen, deLen, startday, trainday, testday, resolution, training_batch, tar):
        print('Preparing data')
        start_time = time.time()
        DC = DataCook()
        DC.data_preprocess(datapath=path, num_zone=1)
        DC.data_roll(enLen, deLen, startday, trainday, testday, resolution)
        DC.data_loader(training_batch, tar)
        self.dataset = DC
        print("--- %s seconds ---" % (time.time() - start_time))

    def Singletrain(self, para):
        self.para = para
        print('Training start')
        start_time_sub = time.time()
        start_time_total = time.time()
        models={}
        for zone in range(self.dataset.num_zone):
            model = OnlySingel.gru_seq2seq(para)
            model = model.cuda()
            model.train_model(dataloder=self.dataset.TrainLoader[zone], zone_index=zone)
            models[zone] = model
            print("{}/{} Finished".format(zone+1, self.dataset.num_zone))
            print("--- %s seconds ---" % (time.time() - start_time_sub))
            start_time_sub = time.time()
            folder_name = "Joule_mdl/" + 'Zone{}'.format(zone)
            mdl_name = 'Train_with_{}days\nTest_on{}.pth'.format(str(self.dataset.trainday), self.dataset.test_start)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            savemodel = os.path.join(folder_name, mdl_name)
            torch.save(model.state_dict(), savemodel)
        self.models = models
        print("--- %s seconds ---" % (time.time() - start_time_total))

    def Singleload(self, para):
        print('Testing start')
        start_time = time.time()
        models = {}
        for zone in range(self.dataset.num_zone):
            model = OnlySingel.gru_seq2seq(para)
            model = model.cuda()
            folder_name = "Joule_mdl/" + 'Zone{}'.format(zone)
            mdl_name = 'Train_with_{}days\nTest_on{}.pth'.format(str(self.dataset.trainday), self.dataset.test_start)
            loadmodel = os.path.join(folder_name, mdl_name)
            model.load_state_dict(torch.load(loadmodel))
            model.eval()
            models[zone] = model
        self.models=models
        print("--- %s seconds ---" % (time.time() - start_time))

    def Singletest(self):
        print('Loading start')
        start_time = time.time()
        for zone in range(self.dataset.num_zone):
            self.models[zone].test_model(dataloder=self.dataset.TestLoader[zone],
                                         tempscal=self.dataset.processed_data[zone]['TzoneScaler'])
        print("--- %s seconds ---" % (time.time() - start_time))

    def Single_Temp_show(self):
        print('Ploting start')
        start_time = time.time()
        rawdf = self.dataset.test_raw_df
        test_len = len(self.models[0].de_denorm)
        pred_len = self.dataset.deLen

        herizon=[1,4,8,12,24]
        MAE, RMSE = {},{}
        for h in herizon:
            aaa = np.array(self.models[0].de_denorm)[:, :4 * h, :].squeeze()
            bbb = rawdf['temp_zone_0']
            aaa = (aaa-32)*5/9
            bbb = (bbb-32)*5/9
            m, r = [], []
            for step in range(aaa.shape[0]):
                m.append(mean_absolute_error(bbb[step:step+h*4], aaa[step]))
                r.append(mean_squared_error(bbb[step:step + h * 4], aaa[step],squared=False))
            MAE[h] = m
            RMSE[h] = r
        MAE = pd.DataFrame.from_dict(MAE)
        RMSE = pd.DataFrame.from_dict(RMSE)
        MAE.to_csv("Temp_mae.csv")
        RMSE.to_csv("Temp_RMSE.csv")


        fig, ax = plt.subplots(1, 1, figsize=(1.96, 1.2), dpi=300, sharex='col', sharey='row', constrained_layout=True)
        sequential_colors = sns.color_palette("RdPu", 5)
        sns.violinplot(data=MAE,palette=sequential_colors, linewidth=0.5,ax=ax, scale="count")
        ax.set_ylabel('MAE (°C)', fontsize=7)
        ax.set_xticklabels(["1hr","4hr","8hr","12hr","24hr"])
        ax.set_ylim(-0.1, 2)
        ax.set_yticks(np.arange(0, 2.1, 0.5))
        ax.set_xlabel('Prediction Horizon', fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7)
        plt.show()
        folder_path = "DynamicModel"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = 'Box_temp[{} to {}].pdf'.format(self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot, transparent=True, format='pdf')

        # MAE = pd.DataFrame.from_dict(MAE)
        # RMSE = pd.DataFrame.from_dict(RMSE)
        # fig, axes = plt.subplots(2, 1, figsize=(4, 3), dpi=300, sharex='col', sharey='row', constrained_layout=True)
        # sequential_colors = sns.color_palette("RdPu", 6)
        # box_plot1 = sns.violinplot(data=MAE,palette=sequential_colors, linewidth=1,ax=axes[0])
        # box_plot2 = sns.violinplot(data=RMSE,palette=sequential_colors, linewidth=1,ax=axes[1])
        # axes[0].set_ylabel('MAE[°C]', fontsize=12, fontweight='bold')
        # axes[1].set_ylabel('RMSE[°C]', fontsize=12, fontweight='bold')
        # axes[1].set_xticklabels(["15min","1hr","4hr","8hr","12hr","24hr"])
        # axes[1].set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
        # axes[0].set_ylim(0.2, 0.8)
        # axes[0].set_yticks(np.arange(0.2, 1, 0.2))
        # axes[1].set_ylim(0.2, 0.8)
        # axes[1].set_yticks(np.arange(0.2, 1, 0.2))
        # ax = box_plot1.axes
        # lines = ax.get_lines()
        # categories = ax.get_xticks()

        # for cat in categories:
        #     # every 4th line at the interval of 6 is median line
        #     # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        #     y = round(lines[4 + cat * 6].get_ydata()[0], 2)
        #     ax.text(
        #         cat,
        #         y,
        #         f'{y}',
        #         ha='center',
        #         va='bottom',
        #         fontweight='bold',
        #         size=6,
        #         color='black')
        #
        # ax = box_plot2.axes
        # lines = ax.get_lines()
        # categories = ax.get_xticks()
        #
        # for cat in categories:
        #     # every 4th line at the interval of 6 is median line
        #     # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        #     y = round(lines[4 + cat * 6].get_ydata()[0], 2)
        #     ax.text(
        #         cat,
        #         y,
        #         f'{y}',
        #         ha='center',
        #         va='bottom',
        #         fontweight='bold',
        #         size=6,
        #         color='black')
        # plt.close(fig)
        # folder_path = "DynamicModel"
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # plotname = 'SBox_Temp[{} to {}].png'.format(self.dataset.test_start, self.dataset.test_end)
        # saveplot = os.path.join(folder_path, plotname)
        # fig.savefig(saveplot)

        fig, ax = plt.subplots(1, 1, figsize=(4.37, 1.3), dpi=300, sharex='col', sharey='row', constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len]-32)*5/9, '-', linewidth=1, color="#159A9C", label='Measurement')
        for timestep in range(test_len)[1::2]:
            tem=self.models[0].de_denorm[timestep][:test_len-timestep]
            tem=(tem-32)*5/9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len-timestep],
                         tem, '--', linewidth=0.5, color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1],((self.models[0].de_denorm[0][0])-32)*5/9, '--', linewidth=1, color="#EA7E7E", label='SeqPINN Prediction')
        ax.legend([],[], frameon=False)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.set_xlabel(None)
        ax.set_ylabel('Temperature (°C)', fontsize=7)
        ax.set_ylim(21, 30)
        ax.set_yticks(np.arange(21, 30, 2))
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
        ax.xaxis.set_major_locator(dates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
        ax.margins(x=0)
        #ax.legend(loc='center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=16, frameon=False)
        plt.show()

        folder_path = "DynamicModel"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plotname = 'Joul_Temp[{} to {}].pdf'.format(self.dataset.test_start, self.dataset.test_end)
        saveplot = os.path.join(folder_path, plotname)
        fig.savefig(saveplot, transparent=True, format='pdf')
        print("--- %s seconds ---" % (time.time() - start_time))

    def Single_save(self):
        print('Saving start')
        start_time = time.time()
        folder_path = "PretrainedModel"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for zone in range(self.dataset.num_zone):
            modelname = 'SSSModel{}[{} to {}].pth'.format(zone, self.dataset.test_start, self.dataset.test_end)
            savemodel = os.path.join(folder_path, modelname)
            torch.save(self.models[zone].state_dict(), savemodel)
        print("--- %s seconds ---" % (time.time() - start_time))



