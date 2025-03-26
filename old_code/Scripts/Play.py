import datetime
from datetime import timedelta
import torch
from tqdm import trange
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from soft_dtw_cuda import SoftDTW
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='../Checkpoint/selected.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, train_loader, valid_loader, test_loader, lr, epochs,
                patience, tempscal, enlen, delen, rawdf, plott, modeltype, scale):
    Simrange={}
    Eplus_start_day = rawdf.index[0] - timedelta(days=1)
    Eplus_end_day = rawdf.index[0] + timedelta(days=1)
    Simrange["start_month"] = Eplus_start_day.strftime("%m")
    Simrange["start_day"] = Eplus_start_day.strftime("%d")
    Simrange["end_month"] = Eplus_end_day.strftime("%m")
    Simrange["end_day"] = Eplus_end_day.strftime("%d")
    with open('../Checkpoint/simrange.pickle', 'wb') as handle:
        pickle.dump(Simrange, handle, protocol=pickle.HIGHEST_PROTOCOL)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    MSE_criterion = nn.MSELoss()
    DTW_criterion = SoftDTW(use_cuda=True, gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    valid_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (trange(epochs) as tr):
        for epoch in tr:
            # Training phase
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                if modeltype == 'SeqPINN':
                    MSE_loss = MSE_criterion(output, target)
                    # DTW_loss = DTW_criterion(output, target).mean() / 20
                    # loss = MSE_loss + DTW_loss * 0
                    loss = MSE_loss
                if modeltype == 'Baseline':
                    MSE_loss = MSE_criterion(output, target[:, enlen:, :])
                    loss = MSE_loss

                loss.backward()
                optimizer.step()
                if modeltype == 'SeqPINN':
                    try:
                        for p in model.decoder_out.parameters():
                            p.data.clamp_(0)
                        for p in model.encoder_out.parameters():
                            p.data.clamp_(0)
                        for p in model.encoder_hvac_linear.parameters():
                            p.data.clamp_(0)
                        for p in model.decoder_hvac_linear.parameters():
                            p.data.clamp_(0)
                    except:
                        0
                else:
                    0
                train_loss += loss.item() * data.size(0)

            # Validation phase
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    if modeltype == 'SeqPINN':
                        MSE_loss = MSE_criterion(output, target)
                        #DTW_loss = DTW_criterion(output, target).mean() / 20
                        #loss = MSE_loss + DTW_loss * 0
                        loss = MSE_loss
                    if modeltype == 'Baseline':
                        MSE_loss = MSE_criterion(output, target[:, enlen:, :])
                        loss = MSE_loss
                    valid_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            #Plot testing results every 20 epoch
            if epoch % 5 == 0:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():  # No gradients needed
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        data_max_check, data_min_check = data.clone(), data.clone()
                        data_max_check[:, enlen:, [7]] =  0  # min cooling
                        data_min_check[:, enlen:, [7]] = -1*scale  # max cooling
                        outputs_max_check = model(data_max_check)
                        outputs_min_check = model(data_min_check)
                        outputs_no_check = model(data)

                outputs_max = outputs_max_check.cpu()
                outputs_min = outputs_min_check.cpu()
                outputs_ = outputs_no_check.cpu()

                outputs_max_denorm, outputs_min_denorm, outputs_denorm = [], [], []

                if modeltype == 'SeqPINN':
                    for idx in range(outputs_max.shape[0]):
                        outputs_max_denorm.append(
                            tempscal.inverse_transform(outputs_max[[idx], enlen:, :].reshape(-1, 1)))
                        outputs_min_denorm.append(
                            tempscal.inverse_transform(outputs_min[[idx], enlen:, :].reshape(-1, 1)))
                        outputs_denorm.append(
                            tempscal.inverse_transform(outputs_[[idx], enlen:, :].reshape(-1, 1)))

                if modeltype == 'Baseline':
                    for idx in range(outputs_max.shape[0]):
                        outputs_max_denorm.append(
                            tempscal.inverse_transform(outputs_max[[idx], :, :].reshape(-1, 1)))
                        outputs_min_denorm.append(
                            tempscal.inverse_transform(outputs_min[[idx], :, :].reshape(-1, 1)))
                        outputs_denorm.append(
                            tempscal.inverse_transform(outputs_[[idx], :, :].reshape(-1, 1)))

                test_len = len(outputs_max_denorm)
                pred_len = delen

                fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8), dpi=300, constrained_layout=True)
                # Plot raw data
                ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9,
                             '-', linewidth=1, color="#159A9C", label='Measurement')
                # Plot PINN
                timestep = 0
                tem = outputs_denorm[timestep][:test_len - timestep]
                tem = (tem - 32) * 5 / 9
                ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                                 tem, '-', linewidth=1, color="gray", label='SeqPINN')

                y_true = ((rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9).reshape(-1, 1)
                y_pred = tem
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred)*100
                ax.text(0.01, 0.8, 'Epochs:{}\nMAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(epoch, mae, mape), fontsize=6, color='gray',
                        fontweight='bold', transform=ax.transAxes)
                # Plot check
                if plott == 'all':
                    timestep = 0
                    minn = outputs_min_denorm[timestep][:test_len - timestep]
                    minn = (minn - 32) * 5 / 9
                    ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], minn, '--', linewidth=1,
                                 color="#8163FD", label='Max Cooling')
                    maxx = outputs_max_denorm[timestep][:test_len - timestep]
                    maxx = (maxx - 32) * 5 / 9
                    ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], maxx, '--', linewidth=1,
                                 color="#FF5F5D", label='Min Cooling')
                else:
                    0

                ax.tick_params(axis='both', which='minor', labelsize=7)
                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.xaxis.set_minor_locator(dates.HourLocator(interval=4))
                ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))
                ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
                ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
                ax.set_xlabel(None)
                ax.set_ylabel('Temperature (°C)', fontsize=7)
                ax.margins(x=0)
                ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False)
                plt.show()
                folder = '../Saved/Training_Image/{}_{}'.format(rawdf.index[0].month, rawdf.index[0].day)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                plot_name = 'Train_with_Epochs{}.png'.format(epoch)
                saveplot = os.path.join(folder, plot_name)
                fig.savefig(saveplot)

            tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
                           valid_loss="{0:.6f}".format(valid_loss))

            # Early Stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('../Checkpoint/selected.pt'))
        return model, train_losses, valid_losses

def test_model(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
    return outputs.cpu()

def check_model(model, test_loader, check_terms, enco):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_max_check, data_min_check = data.clone(), data.clone()
            if check_terms == 'HVAC':
                data_max_check[:, enco:, [7]] =  0  # min cooling
                data_min_check[:, enco:, [7]] = -1  # max cooling
            if check_terms == 'Temp':
                data_max_check[:, enco:, [1]] =  1
                data_min_check[:, enco:, [1]] =  0
            if check_terms == 'Solar':
                data_max_check[:, enco:, [2]] =  1
                data_min_check[:, enco:, [2]] =  0
            outputs_max_check = model(data_max_check)
            outputs_min_check = model(data_min_check)
            outputs_no_check = model(data)
    return outputs_max_check.cpu(), outputs_min_check.cpu(), outputs_no_check.cpu()
