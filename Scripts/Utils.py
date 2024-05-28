from Dataset import DataCook
from Config import paras
import pickle
import os
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from Models import SeqPINN
from Play import train_model, test_model
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

class ddpred:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.para = None
        self.args = None

    def data_ready(self, args):
        print('Preparing data')
        start_time = time.time()
        DC = DataCook()
        DC.data_preprocess(datapath=args.path, num_zone=1)
        DC.data_roll(args=args)
        DC.data_loader(args.training_batch)
        para = paras(args=args)
        self.dataset = DC
        self.para = para
        print("--- %s seconds ---" % (time.time() - start_time))

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SeqPINN.SeqPinn(self.para).to(device)
        start_time = time.time()
        print('Trainging')
        self.model, self.train_losses, self.valid_losses = train_model(model=model,
                                                                       train_loader=self.dataset.TrainLoader[0],
                                                                       valid_loader=self.dataset.ValidLoader[0],
                                                                       lr=self.para['lr'],
                                                                       epochs=self.para['epochs'],
                                                                       patience=self.para['patience'])
        print("--- %s seconds ---" % (time.time() - start_time))
        print('Saving start')

        folder_name = "../Saved/Trained_mdl" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        mdl_name = 'Train_with_{}days\nTest_on{}.pth'.format(str(self.dataset.trainday), self.dataset.test_start)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savemodel = os.path.join(folder_name, mdl_name)
        torch.save(model.state_dict(), savemodel)

    def train_valid_loss_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(1.96, 1.96), dpi=300, constrained_layout=True)
        ax.plot(self.train_losses, label='Training_loss', color='#F0907F')
        ax.plot(self.valid_losses, label='Validation loss', color='#8B90F5')
        ax.set_ylabel('Loss', fontsize=7)
        ax.set_xlabel('Epochs', fontsize=7)
        ax.tick_params(axis='y', which='both', labelsize=7, pad=0.7)
        ax.tick_params(axis='x', which='both', labelsize=7, pad=0.7)
        ax.set_ylim(0, 0.1)
        # Creating an inset of the size [width, height] starting at position (x, y)
        # all quantities are in fraction of figure width or height
        axins = inset_axes(ax, width="120%", height="120%", loc='upper right',
                           bbox_to_anchor=(0.5, 0.5, 0.45, 0.45),
                           bbox_transform=ax.transAxes)

        # Plot the same data on the inset
        axins.plot(self.train_losses, label='Training_loss', color='#F0907F')
        axins.plot(self.valid_losses, label='Validation loss', color='#8B90F5')

        # Specify the limits of your zoom-in area
        x1, x2, y1, y2 = 50, 120, 0.001, 0.008  # These limits should be set according to your data
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # Optionally add grid, customize ticks, etc.
        axins.grid(True)
        axins.tick_params(axis='both', which='both', labelsize=5)
        ax.legend(loc='center', bbox_to_anchor=(0.5, 1.13), ncol=1, fontsize=7, frameon=False)
        plt.show()

    def load(self):
        print('Loading start')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start_time = time.time()
        model = SeqPINN.SeqPinn(self.para).to(device)
        folder_name = "../Saved/Trained_mdl" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
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

        outputs = test_model(model=self.model, test_loader=self.dataset.TestLoader[0])
        # De-norm
        to_out, en_out, de_out = [], [], []
        tempscal = self.dataset.processed_data[0]['TzoneScaler']
        for idx in range(outputs.shape[0]):
            to_out.append(tempscal.inverse_transform(outputs[[idx], :, :].reshape(-1, 1)))
            en_out.append(tempscal.inverse_transform(outputs[[idx], :self.para['encoLen'], :].reshape(-1, 1)))
            de_out.append(tempscal.inverse_transform(outputs[[idx], self.para['encoLen']:, :].reshape(-1, 1)))
        self.to_out = to_out
        self.en_out = en_out
        self.de_out = de_out

        folder_name = "../Result/" + 'Enco{}_Deco{}'.format(str(self.dataset.enLen), str(self.dataset.deLen))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_name = '(raw)Train_with_{}days\nTest_on{}.csv'.format(str(self.dataset.trainday), self.dataset.test_start)
        savecsv = os.path.join(folder_name, csv_name)
        self.dataset.test_raw_df.to_csv(savecsv)
        test_len = len(self.de_out)
        pred_len = self.dataset.deLen
        test_result = {}
        mae = {}
        mape = {}
        gdtruth = (self.dataset.test_raw_df['temp_zone_{}'.format(0)].values - 32) * 5 / 9
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
        print("--- %s seconds ---" % (time.time() - start_time))

    def prediction_show(self):
        print('Ploting start')
        rawdf = self.dataset.test_raw_df
        test_len = len(self.de_out)
        pred_len = self.dataset.deLen
        fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=300, constrained_layout=True)
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9, '-',
                     linewidth=1.5, color="#159A9C", label='Measurement')
        mae = np.array(list(self.mae.values())).mean()
        mape = np.array(list(self.mape.values())).mean()
        for timestep in range(test_len):
            tem = self.de_out[timestep][:test_len - timestep]
            tem = (tem - 32) * 5 / 9
            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                         tem, '--', linewidth=0.4, color="#EA7E7E")
        ax.plot_date(rawdf.index[0:0 + 1], ((self.de_out[0][0]) - 32) * 5 / 9, '--', linewidth=1.5,
                     color="#EA7E7E", label='SeqPINN Prediction')
        ax.text(0.01, 0.85, 'MAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(mae, mape), fontsize=10, color='gray',
                fontweight='bold', transform=ax.transAxes)
        ax.legend([], [], frameon=False)
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
