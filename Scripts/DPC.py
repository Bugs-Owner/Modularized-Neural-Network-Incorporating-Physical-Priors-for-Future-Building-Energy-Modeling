import torch
from tqdm import trange
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def Online_solve(control_mdl, dynamic_mdl, loader, args):
    optimizer = torch.optim.Adam(control_mdl.parameters(), lr=args.dpclr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_scalers = joblib.load('../Checkpoint/saved_scalers.pkl')
    Tscale = torch.from_numpy(saved_scalers['TzoneScaler'].scale_).to(device)
    Tmin = torch.from_numpy(saved_scalers['TzoneScaler'].min_).to(device)
    Uscale = torch.from_numpy(saved_scalers['PhvacScaler'].scale_).to(device)
    Umin = torch.from_numpy(saved_scalers['PhvacScaler'].min_).to(device)
    with (trange(args.dpcepochs) as tr):
        for epoch in tr:
            # Training phase
            control_mdl.train()
            train_losses = []
            train_loss,temperature_loss,pHVAC_loss = 0,0,0

            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                u = control_mdl(data)
                data_copy = data.clone()
                data_copy[:,-96:,[7]]=u[:,-96:,:]
                t = dynamic_mdl(data_copy)
                p = data_copy[:, :, [11]]
                t_norm = (t - Tmin) / Tscale
                upper_norm = (data_copy[:, :, [8]] - Tmin) / Tscale
                lower_norm = (data_copy[:, :, [9]] - Tmin) / Tscale
                u_norm = (u - Umin) / Uscale / 1000

                uppervio = nn.functional.relu(t_norm - upper_norm)
                lowervio = nn.functional.relu(lower_norm - t_norm)

                U_loss_ = torch.sum(torch.square(u_norm*p))
                T_loss_ = (torch.mean(torch.square(uppervio)) + torch.mean(torch.square(lowervio))) * 10000

                to_loss = U_loss_ + T_loss_
                to_loss.backward()
                optimizer.step()

                train_loss += to_loss.item() * data.size(0)
                temperature_loss += T_loss_.item() * data.size(0)
                pHVAC_loss += U_loss_.item() * data.size(0)

            train_loss = train_loss / len(loader.dataset)
            temperature_loss = temperature_loss / len(loader.dataset)
            pHVAC_loss = pHVAC_loss / len(loader.dataset)

            train_losses.append(train_loss)
            tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
                           temperature_loss="{0:.6f}".format(temperature_loss),pHVAC_loss="{0:.6f}".format(pHVAC_loss))
    return u_norm * 1000 * -1
        #
        #
        #     #Plot testing results every 20 epoch
        #     if epoch % 5 == 0:
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         with torch.no_grad():  # No gradients needed
        #             for data, target in test_loader:
        #                 data, target = data.to(device), target.to(device)
        #                 data_max_check, data_min_check = data.clone(), data.clone()
        #                 data_max_check[:, enlen:, [7]] = 0   # min cooling
        #                 data_min_check[:, enlen:, [7]] = -5  # max cooling
        #                 outputs_max_check = model(data_max_check)
        #                 outputs_min_check = model(data_min_check)
        #                 outputs_no_check = model(data)
        #
        #         outputs_max = outputs_max_check.cpu()
        #         outputs_min = outputs_min_check.cpu()
        #         outputs_ = outputs_no_check.cpu()
        #
        #         outputs_max_denorm, outputs_min_denorm, outputs_denorm = [], [], []
        #
        #         if modeltype == 'SeqPINN':
        #             for idx in range(outputs_max.shape[0]):
        #                 outputs_max_denorm.append(
        #                     tempscal.inverse_transform(outputs_max[[idx], enlen:, :].reshape(-1, 1)))
        #                 outputs_min_denorm.append(
        #                     tempscal.inverse_transform(outputs_min[[idx], enlen:, :].reshape(-1, 1)))
        #                 outputs_denorm.append(
        #                     tempscal.inverse_transform(outputs_[[idx], enlen:, :].reshape(-1, 1)))
        #
        #         if modeltype == 'Baseline':
        #             for idx in range(outputs_max.shape[0]):
        #                 outputs_max_denorm.append(
        #                     tempscal.inverse_transform(outputs_max[[idx], :, :].reshape(-1, 1)))
        #                 outputs_min_denorm.append(
        #                     tempscal.inverse_transform(outputs_min[[idx], :, :].reshape(-1, 1)))
        #                 outputs_denorm.append(
        #                     tempscal.inverse_transform(outputs_[[idx], :, :].reshape(-1, 1)))
        #
        #         test_len = len(outputs_max_denorm)
        #         pred_len = delen
        #
        #         fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.8), dpi=300, constrained_layout=True)
        #         # Plot raw data
        #         ax.plot_date(rawdf.index[:test_len], (rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9,
        #                      '-', linewidth=1, color="#159A9C", label='Measurement')
        #         # Plot PINN
        #         timestep = 0
        #         tem = outputs_denorm[timestep][:test_len - timestep]
        #         tem = (tem - 32) * 5 / 9
        #         ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep],
        #                          tem, '-', linewidth=1, color="gray", label='SeqPINN')
        #
        #         y_true = ((rawdf['temp_zone_{}'.format(0)].values[:test_len] - 32) * 5 / 9).reshape(-1, 1)
        #         y_pred = tem
        #         mae = mean_absolute_error(y_true, y_pred)
        #         mape = mean_absolute_percentage_error(y_true, y_pred)
        #         ax.text(0.01, 0.8, 'Epochs:{}\nMAE:{:.2f}[°C]\nMAPE:{:.2f}[%]'.format(epoch, mae, mape), fontsize=6, color='gray',
        #                 fontweight='bold', transform=ax.transAxes)
        #         # Plot check
        #         if plott == 'all':
        #             timestep = 0
        #             minn = outputs_min_denorm[timestep][:test_len - timestep]
        #             minn = (minn - 32) * 5 / 9
        #             ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], minn, '--', linewidth=1,
        #                          color="#8163FD", label='Max Cooling')
        #             maxx = outputs_max_denorm[timestep][:test_len - timestep]
        #             maxx = (maxx - 32) * 5 / 9
        #             ax.plot_date(rawdf.index[timestep:timestep + pred_len][:test_len - timestep], maxx, '--', linewidth=1,
        #                          color="#FF5F5D", label='Min Cooling')
        #         else:
        #             0
        #
        #         ax.tick_params(axis='both', which='minor', labelsize=7)
        #         ax.tick_params(axis='both', which='major', labelsize=7)
        #         ax.xaxis.set_minor_locator(dates.HourLocator(interval=4))
        #         ax.xaxis.set_minor_formatter(dates.DateFormatter("%H"))
        #         ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        #         ax.xaxis.set_major_formatter(dates.DateFormatter('%b-%d'))
        #         ax.set_xlabel(None)
        #         ax.set_ylabel('Temperature (°C)', fontsize=7)
        #         ax.margins(x=0)
        #         ax.legend(loc='center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=6, frameon=False)
        #         plt.show()
        #         folder = '../Saved/Training_Image/{}_{}'.format(rawdf.index[0].month, rawdf.index[0].day)
        #         if not os.path.exists(folder):
        #             os.makedirs(folder)
        #         plot_name = 'Train_with_Epochs{}.png'.format(epoch)
        #         saveplot = os.path.join(folder, plot_name)
        #         fig.savefig(saveplot)
        #
        #     tr.set_postfix(epoch="{0:.0f}".format(epoch+1), train_loss="{0:.6f}".format(train_loss),
        #                    valid_loss="{0:.6f}".format(valid_loss))
        #
        #     # Early Stopping
        #     early_stopping(valid_loss, model)
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
        #
        # # load the last checkpoint with the best model
        # model.load_state_dict(torch.load('../Checkpoint/selected.pt'))
        # return model, train_losses, valid_losses

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

def optimize(building_mdl, control_mdl, data, lr, epochs, patience, tempscal, enlen, delen, rawdf, plott):
    optimizer = torch.optim.Adam(control_mdl.parameters(), lr=lr)
    MSE_criterion = nn.MSELoss()
    DTW_criterion = SoftDTW(use_cuda=True, gamma=0.1)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    train_losses = []
    valid_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (trange(epochs) as tr):
        for epoch in tr:
            # Online Solve
            control_mdl.train()
            train_loss = 0.0
            data = data.to(device)
            optimizer.zero_grad()
            u_opt = control_mdl(data)



            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                if modeltype == 'SeqPINN':
                    MSE_loss = MSE_criterion(output, target)
                    DTW_loss = DTW_criterion(output, target).mean() / 20
                    loss = MSE_loss + DTW_loss * 0
                if modeltype == 'Baseline':
                    MSE_loss = MSE_criterion(output, target[:, enlen:, :])
                    loss = MSE_loss

                loss.backward()
                optimizer.step()
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
                        DTW_loss = DTW_criterion(output, target).mean() / 20
                        loss = MSE_loss + DTW_loss * 0
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
                        data_max_check[:, enlen:, [7]] = 0   # min cooling
                        data_min_check[:, enlen:, [7]] = -1  # max cooling
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
                mape = mean_absolute_percentage_error(y_true, y_pred)
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
                folder = '../Saved/Trainging_Image'
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