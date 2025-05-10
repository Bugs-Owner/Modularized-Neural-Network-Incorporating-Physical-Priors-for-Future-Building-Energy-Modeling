import numpy as np
import torch
from tqdm import trange
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import time
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def calc_correlation(actual, predic):
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='../Checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
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

        folder_name = self.path
        mdl_name = 'best.pt'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        savemodel = os.path.join(folder_name, mdl_name)
        self.savemodel = savemodel
        torch.save(model.state_dict(), savemodel)

        self.val_loss_min = val_loss

def train_model(model, train_loader, valid_loader, test_loader, lr, epochs,
                patience, tempscal, fluxscal, enlen, delen,
                rawdf, plott, modeltype, scale, device, ext_mdl):
    num_update = 0
    total_time = 0
    time_every_update = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    MSE_criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_total_losses, train_temp_losses, train_diff_losses = [], [], []
    valid_losses = []
    vio_positive_loss = []
    vio_negative_loss = []
    mae_loss = []
    device = device

    diff_alpha = 0.3  # weight for temporal difference loss

    # Flags for physics-informed loss
    Phy_loss, Phy_cons = {
        "LSTM": (0, 0),
        "PI-modnn": (1, 1),
        "PI-modnn|C": (1, 0),
        "PI-modnn|L": (0, 1),
        "PI-modnn|LC": (0, 0)
    }.get(modeltype, (None, None))
    if envelop_mdl=="physics":
        Phy_loss, Phy_cons = None, None
    if Phy_loss is None:
        raise ValueError("Unknown modeltype")

    with (trange(epochs) as tr):
        for epoch in tr:
            # Training phase
            model.train()
            train_loss_temp  = 0.0
            train_loss_diff  = 0.0
            train_loss_total = 0.0
            for data, target in train_loader:
                time_start = time.time()
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                # In real world application, we perhaps only have label for Tzone
                # So we only compute loss based on that
                # But if the flux is avaliable, like EPlus data, we can add it here to improve model performance
                Temp,_,_ = model(data)
                Temp_loss = MSE_criterion(Temp[:, enlen:, :], target[:, enlen:, [0]])
                Diff_loss = torch.mean(torch.abs((Temp[:, 1:, :] - Temp[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])))

                loss = Temp_loss + diff_alpha * Diff_loss * Phy_loss
                loss.backward()
                optimizer.step()

                if Phy_cons==1:
                    try:
                        # Positive Hard Constraints
                        model.Zone.scale.weight.data.clamp_(0)
                        model.Int.scale.weight.data.clamp_(0)
                        model.HVAC.scale.weight.data.clamp_(0)
                        model.Ext.conduction.weight.data.clamp_(0)
                        if ext_mdl == 'RNN':
                            model.Ext.rnn.weight.data.clamp_(0)
                        else:
                            pass
                    except:
                        pass
                time_elapsed = time.time() - time_start
                total_time += time_elapsed
                num_update += 1
                time_every_update[num_update] = time_elapsed
                train_loss_temp += Temp_loss.item() * data.size(0)
                train_loss_diff += Diff_loss.item() * data.size(0)
                train_loss_total += loss.item() * data.size(0)

            # fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
            # plt.plot(Temp.cpu().detach().numpy()[0, :, :])
            # plt.plot(target[:, :, [0]].cpu().detach().numpy()[0, :, :])
            # plt.show()
            # Validation phase
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    Temp,_,_ = model(data)
                    Temp_loss = MSE_criterion(Temp[:, enlen:, :], target[:, enlen:, [0]])
                    Diff_loss = torch.mean(
                        torch.abs((Temp[:, 1:, :] - Temp[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])))
                    loss = Temp_loss + diff_alpha * Diff_loss * Phy_loss
                    valid_loss += loss.item() * data.size(0)

            train_loss_total = train_loss_total / len(train_loader.dataset)
            Temp_loss = train_loss_temp / len(train_loader.dataset)
            Diff_loss = train_loss_diff / len(train_loader.dataset)
            valid_loss = valid_loss / len(valid_loader.dataset)

            train_total_losses.append(train_loss_total)
            train_temp_losses.append(Temp_loss)
            train_diff_losses.append(Diff_loss)
            valid_losses.append(valid_loss)

            res=10
            # How many epoch we want to display our training results
            # It can help us to dynamically understand the model performance
            if epoch % res == 0:
                with torch.no_grad():  # No gradients needed
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        data_max_check, data_min_check = data.clone(), data.clone()
                        # we check the model response under max heating and cooling
                        data_max_check[:, enlen-1:, [6]] = 1*scale   # max heating
                        data_min_check[:, enlen-1:, [6]] = -1*scale  # max cooling

                        outputs_max_check, hvac_max_check, other_max_check = model(data_max_check)
                        outputs_min_check, hvac_min_check, other_min_check = model(data_min_check)
                        Temp, hvac_flux, other_flux = model(data)

                outputs_max, hvac_max, other_max = outputs_max_check.cpu(), hvac_max_check.cpu(), other_max_check
                outputs_min, hvac_min, other_min = outputs_min_check.cpu(), hvac_min_check.cpu(), other_min_check
                Temp_out, hvac_out, other_out = Temp.cpu(), hvac_flux.cpu(), other_flux
                external_out = other_out[0].cpu()
                inter_out = other_out[1].cpu()
                delta_out = other_out[2].cpu()

                outputs_max_denorm, outputs_min_denorm = [], []
                Temp_out_denorm, Other_out_denorm = [], []
                hvac_max_denorm, hvac_min_denorm, hvac_out_denorm = [], [], []
                inter_out_denorm, external_out_denorm, delta_out_denorm = [], [], []

                for idx in range(outputs_max.shape[0]):
                    outputs_max_denorm.append(
                        tempscal.inverse_transform(outputs_max[[idx], enlen:, :].reshape(-1, 1)))
                    outputs_min_denorm.append(
                        tempscal.inverse_transform(outputs_min[[idx], enlen:, :].reshape(-1, 1)))
                    Temp_out_denorm.append(
                        tempscal.inverse_transform(Temp_out[[idx], enlen:, :].reshape(-1, 1)))

                    hvac_max_denorm.append(
                        fluxscal.inverse_transform(hvac_max[[idx], enlen:, :].reshape(-1, 1)))
                    hvac_min_denorm.append(
                        fluxscal.inverse_transform(hvac_min[[idx], enlen:, :].reshape(-1, 1)))
                    hvac_out_denorm.append(
                        fluxscal.inverse_transform(hvac_out[[idx], enlen:, :].reshape(-1, 1)))

                    #TODO: collect different heat fluxes, need match with different model, leave it  later
                    # inter_out_denorm.append(
                    #     fluxscal.inverse_transform(inter_out[[idx], enlen:, :].reshape(-1, 1)))
                    # external_out_denorm.append(
                    #     fluxscal.inverse_transform(external_out[[idx], enlen:, :].reshape(-1, 1)))
                    # delta_out_denorm.append(
                    #     fluxscal.inverse_transform(delta_out[[idx], enlen:, :].reshape(-1, 1)))

                test_len = len(outputs_max_denorm)
                pred_len = delen
                display = True
                if display == True :
                    fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
                    # Plot raw data
                    ax.plot_date(rawdf.index[:test_len], (rawdf['temp_room'].values[:test_len] - 32) * 5 / 9,
                                 '-', linewidth=1, color="#159A9C", label='Measurement')
                    # Plot
                    name = modeltype
                    timestep = 0
                    tem = Temp_out_denorm[timestep][:test_len - timestep]
                    tem = (tem - 32) * 5 / 9
                    ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
                                     tem, '-', linewidth=1, color="gray", label=name)

                    # Plot check
                    if plott == 'all':
                        for timestep in [0]:
                            minn = outputs_min_denorm[timestep][:test_len - timestep]
                            minn = (minn - 32) * 5 / 9
                            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], minn, '--', linewidth=1,
                                         color="#8163FD", label='Max Cooling')
                            maxx = outputs_max_denorm[timestep][:test_len - timestep]
                            maxx = (maxx - 32) * 5 / 9
                            ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], maxx, '--', linewidth=1,
                                         color="#FF5F5D", label='Max Heating')
                    else:
                        pass

                    y_true = ((rawdf['temp_room'].values[:delen][:test_len] - 32) * 5 / 9).reshape(-1, 1)
                    y_pred = tem
                    mae = mean_absolute_error(y_true, y_pred)
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                    r2 = calc_correlation(y_true, y_pred)

                    vio1, vio2 = 0, 0
                    # Temperature response violation
                    for timestep in range(len(Temp_out_denorm)):
                        minn = outputs_min_denorm[timestep][:test_len - timestep]
                        minn = (minn - 32) * 5 / 9
                        maxx = outputs_max_denorm[timestep][:test_len - timestep]
                        maxx = (maxx - 32) * 5 / 9
                        y_pred = (Temp_out_denorm[timestep][:test_len - timestep] - 32) * 5 / 9
                        y_pred = y_pred
                        vio1 += np.clip((minn - y_pred), 0, None).sum()
                        vio2 += np.clip((y_pred - maxx), 0, None).sum()

                    vio_positive_loss.append(vio2)
                    vio_negative_loss.append(vio1)
                    mae_loss.append(mae)
                    text_gray = 'Epochs:{}\nMAE:{:.2f}[°C]\nMAPE:{:.2f}[%]\nR2:{:.2f}\n'.format(epoch, mae,
                                                                                                            mape, r2)
                    # Add the gray text
                    ax.text(1.01, 0.55, text_gray, fontsize=6, color='gray', fontweight='bold', transform=ax.transAxes)
                    text_red = 'TRV+:{:.1f}[°C-h]'.format(vio2)
                    text_blue = 'TRV-:{:.1f}[°C-h]'.format(vio1)
                    ax.text(1.01, 0.15, text_red, fontsize=6, color='red', transform=ax.transAxes)
                    ax.text(1.01, 0.01, text_blue, fontsize=6, color='blue', transform=ax.transAxes)

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
                    # folder = '../Saved/Training_Image/{}/{}_{}'.format(modeltype, rawdf.index[0].month, rawdf.index[0].day)
                    # if not os.path.exists(folder):
                    #     os.makedirs(folder)
                    # plot_name = 'Train_with_Epochs{}.png'.format(epoch)
                    # saveplot = os.path.join(folder, plot_name)
                    # fig.savefig(saveplot)
                flux = False
                # The proposed model can not only learn the temperature
                # But also lean the heat flux, change it to yes to see flux response
                # However, in most cases, we don't have flux observation, so this output is hard to guarantee and evaluate
                if flux == True :
                    fig, axs = plt.subplots(4, 1, figsize=(3.2, 4.8), dpi=300, constrained_layout=True, sharex=True)
                    timestep = 0
                    inter = inter_out_denorm[timestep][:test_len - timestep]
                    outer = external_out_denorm[timestep][:test_len - timestep]
                    hvac = hvac_out_denorm[timestep][:test_len - timestep]
                    total = delta_out_denorm[timestep][:test_len - timestep]
                    time_index = rawdf.index[timestep:timestep + pred_len][:test_len - timestep]

                    # --- Plot Temp ---
                    axs[0].plot_date(time_index, tem, '--', linewidth=1, color="green", label="Temp")
                    axs[0].plot_date(time_index, (rawdf["temp_room"][timestep:timestep + pred_len][:test_len - timestep]-32)*5/9,
                                     '-', linewidth=1, color="green", label="Temp Mear")
                    axs[0].set_ylabel('Zone Temp\n(°C)', fontsize=7)
                    axs[0].legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6, frameon=False)
                    axs[0].tick_params(axis='both', which='both', labelsize=7)

                    # --- Plot IntGain ---
                    axs[1].plot_date(time_index, inter, '--', linewidth=1, color="#8163FD", label="IntGain")
                    axs[1].plot_date(time_index, rawdf["Int_cov"][timestep:timestep + pred_len][:test_len - timestep],
                                     '-', linewidth=1, color="#8163FD", label="IntGain Mear")
                    axs[1].set_ylabel('Heat Transfer\n(W)', fontsize=7)
                    axs[1].legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6, frameon=False)
                    axs[1].tick_params(axis='both', which='both', labelsize=7)

                    # --- Plot ExtGain ---
                    axs[2].plot_date(time_index, outer, '--', linewidth=1, color="#FF5F5D", label="ExtGain")
                    axs[2].plot_date(time_index, rawdf["Sur_cov"][timestep:timestep + pred_len][:test_len - timestep],
                                     '-', linewidth=1, color="#FF5F5D", label="ExtGain Mear")
                    axs[2].set_ylabel('Heat Transfer\n(W)', fontsize=7)
                    axs[2].legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6, frameon=False)
                    axs[2].tick_params(axis='both', which='both', labelsize=7)

                    # --- Plot Delta Q ---
                    axs[3].plot_date(time_index, total, '--', linewidth=1, color="black", label="Delta Q")
                    axs[3].plot_date(time_index, rawdf["Eng_stg"][timestep:timestep + pred_len][:test_len - timestep],
                                     '-', linewidth=1, color="black", label="Delta Q Mear")
                    axs[3].set_ylabel('Heat Transfer\n(W)', fontsize=7)
                    axs[3].legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=6, frameon=False)
                    axs[3].tick_params(axis='both', which='both', labelsize=7)

                    # Set X axis for bottom subplot only
                    axs[3].xaxis.set_minor_locator(dates.HourLocator(interval=4))
                    axs[3].xaxis.set_minor_formatter(dates.DateFormatter("%H"))
                    axs[3].xaxis.set_major_locator(dates.DayLocator(interval=1))
                    axs[3].xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
                    axs[3].set_xlabel(None)
                    axs[3].margins(x=0)

                    plt.show()
                else:
                    pass

            tr.set_postfix(epoch="{0:.0f}".format(epoch+1),
                           train_loss_total="{0:.5f}".format(train_loss_total),
                           train_loss_temp="{0:.5f}".format(Temp_loss),
                           train_loss_diff="{0:.5f}".format(Diff_loss),
                           valid_loss="{0:.5f}".format(valid_loss))

            # Early Stopping
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load("../Checkpoint/best.pt"))

        train_log = {
            'train_total_losses': train_total_losses,
            'train_temp_losses': train_temp_losses,
            'train_diff_losses': train_diff_losses,
            'valid_losses': valid_losses,
            'vio_positive_loss': vio_positive_loss,
            'vio_negative_loss': vio_negative_loss,
            'mae_loss': mae_loss,
            'total_time': total_time,
            'num_update': num_update,
            'time_every_update': time_every_update
        }

        return model, train_log

def eval_model(model, test_loader, modeltype, tempscal, enlen, rawdf, device):
    # evaluate one-step prediction
    # no need to run
    model.eval()
    device = device
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred_list = []
            for step in range(data.shape[0]):
                input = data.clone()
                if step > 0:
                    #overwrite based on prediction
                    input[step,enlen-min(enlen,step):enlen,0] = torch.tensor(pred_list).to(device)[-min(enlen,step):]
                    pred_1step = model(input[[step],:,:]).squeeze()[enlen:enlen + 1]
                    pred_list.append(pred_1step)
                else:
                    pred_1step = model(input[[step],:,:]).squeeze()[enlen:enlen + 1]
                    pred_list.append(pred_1step)
                # pred_1step = model(data[[step], :, :]).squeeze()[enlen:enlen + 1]
                # pred_list.append(pred_1step)
        eva_out = torch.tensor(pred_list).cpu()
        denorm_eva_out = tempscal.inverse_transform(eva_out.reshape(-1, 1))

        test_len = len(denorm_eva_out)
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.8), dpi=300, constrained_layout=True)
        # Plot raw data
        ax.plot_date(rawdf.index[:test_len], (rawdf['temp_room'].values[:test_len] - 32) * 5 / 9,
                     '-', linewidth=1, color="#159A9C", label='Measurement')
        # Plot
        if modeltype == 'SeqPINN' or modeltype == 'modnn':
            name = 'modnn'
        else:
            name = 'LSTM'

        tem = denorm_eva_out
        tem = (tem - 32) * 5 / 9
        ax.plot_date(rawdf.index[:test_len],
                     tem, '-', linewidth=1, color="gray", label=name)

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

def test_model(model, test_loader, device, tempscal, enlen):
    model.eval()
    device = device
    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            temp,_,_ = model(data)
    temp=temp.cpu()
    to_out, en_out, de_out = [], [], []
    for idx in range(temp.shape[0]):
        to_out.append(tempscal.inverse_transform(temp[[idx], :, :].reshape(-1, 1)))
        en_out.append(tempscal.inverse_transform(temp[[idx], :enlen, :].reshape(-1, 1)))
        de_out.append(tempscal.inverse_transform(temp[[idx], enlen:, :].reshape(-1, 1)))
    to_out = to_out
    en_out = en_out
    de_out = de_out

    return to_out, en_out, de_out

def check_model(model, test_loader, enlen, scale, tempscal, hvacscale, checkscale, device):
    model.eval()
    device = device
    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_max_check, data_min_check = data.clone(), data.clone()
            # Sanity check
            # Can model response well with max heating and cooling?
            data_max_check[:, enlen-1:, [6]] = 1 * scale * checkscale  # max heating
            data_min_check[:, enlen-1:, [6]] = -1 * scale * checkscale  # max cooling

            outputs_max_check, _, _= model(data_max_check)
            outputs_min_check, _, _= model(data_min_check)
            outputs_no_check, _, _= model(data)

    outputs_max = outputs_max_check.cpu()
    outputs_min = outputs_min_check.cpu()
    outputs_ = outputs_no_check.cpu()

    outputs_max_denorm, outputs_min_denorm, outputs_denorm = [], [], []
    target_debug = []
    cooling_max_denorm, cooling_min_denorm = [], []

    for idx in range(outputs_max.shape[0]):
        outputs_max_denorm.append(
            tempscal.inverse_transform(outputs_max[[idx], enlen:, :].reshape(-1, 1)))
        outputs_min_denorm.append(
            tempscal.inverse_transform(outputs_min[[idx], enlen:, :].reshape(-1, 1)))
        outputs_denorm.append(
            tempscal.inverse_transform(outputs_[[idx], enlen:, :].reshape(-1, 1)))
        target_debug.append(
            tempscal.inverse_transform(target.cpu()[[idx], enlen:, :].reshape(-1, 1)))
        cooling_max_denorm.append(
            hvacscale.inverse_transform(data_max_check.cpu()[[idx], enlen:, [6]].reshape(-1, 1)))
        cooling_min_denorm.append(
            hvacscale.inverse_transform(data_min_check.cpu()[[idx], enlen:, [6]].reshape(-1, 1)))

    print(outputs_max_denorm[0][0], outputs_min_denorm[0][0], outputs_denorm[0][0])
    return outputs_max_denorm, outputs_min_denorm, outputs_denorm, cooling_max_denorm, cooling_min_denorm

def dynamic_check_model(model, test_loader, enlen, scale, tempscal, hvacscale, device):
    model.eval()
    dynamic_temp = {}
    dynamic_hvac = {}
    # Same idea, can model response well under changing control input?
    # You can adjust the range, but please update the color map accordingly
    for u in [-4000,-2000,0,2000,4000]:
        range_min, range_max = hvacscale.feature_range
        data_min, data_max = hvacscale.data_min_, hvacscale.data_max_
        u_norm = (u - data_min) / (data_max - data_min) * (range_max-range_min) + range_min
        u_norm = torch.from_numpy(u_norm).float().to(device)
        with torch.no_grad():  # No gradients needed
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data_check = data.clone()
                data_check[:, enlen - 1:, [6]] = u_norm * scale

                outputs_no_check,_,_ = model(data)
                dynamic_check,_,_ = model(data_check)

        dynamic_check = dynamic_check.cpu()
        outputs_ = outputs_no_check.cpu()

        outputs_check_denorm, outputs_denorm, phvac_denorm = [], [], []

        for idx in range(dynamic_check.shape[0]):
            outputs_check_denorm.append(
                tempscal.inverse_transform(dynamic_check[[idx], enlen:, :].reshape(-1, 1)))
            outputs_denorm.append(
                tempscal.inverse_transform(outputs_[[idx], enlen:, :].reshape(-1, 1)))
            phvac_denorm.append(
                hvacscale.inverse_transform(data_check.cpu()[[idx], enlen:, [6]].reshape(-1, 1)))

        dynamic_temp[u] = outputs_check_denorm
        dynamic_hvac[u] = phvac_denorm

    return dynamic_temp, dynamic_hvac

def grad_model(model, test_loader, device):
    from torch.autograd.functional import jacobian
    model.train()

    def model_output(x):
        output, _, _ = model(x)
        return output

    for data, target in test_loader:
        Joc_input, _ = data.to(device), target.to(device)
        Joc_input.requires_grad_(True)

        Joc_matrix_list = []
        for index in range(Joc_input.shape[0]):
            Joc_matrix = jacobian(model_output, Joc_input[[index], :, :])
            Joc_matrix_list.append(Joc_matrix.detach().cpu().numpy().squeeze())

    return Joc_matrix_list


