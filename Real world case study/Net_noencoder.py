import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
from Solver import NNsolver

class gru_Linear(nn.Module):

    def __init__(self, input_size, h1, h2, output_size):
        super(gru_Linear, self).__init__()
        self.Fc1 = nn.Linear(in_features=input_size, out_features=h1, bias=True)
        self.Fc2 = nn.Linear(in_features=h1, out_features=h2, bias=True)
        self.Fc3 = nn.Linear(in_features=h2, out_features=output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x_mix):
        Embedding_state = self.Fc1(x_mix)
        Embedding_state = self.relu(Embedding_state)
        Embedding_state = self.Fc2(Embedding_state)
        Embedding_state = self.relu(Embedding_state)
        Embedding_state = self.Fc3(Embedding_state)

        return Embedding_state

class gru_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(gru_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bias=True)
        self.Defc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x_input, hidden_states):
        gru_out, self.hidden = self.gru(x_input, hidden_states)
        output = self.Defc(gru_out)

        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class gru_seq2seq(nn.Module):

    def __init__(self, para):

        super(gru_seq2seq, self).__init__()
        self.decoder_external_in, self.decoder_external_h, self.decoder_external_out = para["decoder_external_in"], \
            para["decoder_external_h"], \
            para["decoder_external_out"]
        self.decoder_internal_in, self.decoder_internal_h, self.decoder_internal_out = para["decoder_internal_in"], \
            para["decoder_internal_h"], \
            para["decoder_internal_out"]
        self.decoder_hvac_in, self.decoder_hvac_h, self.decoder_hvac_out = para["decoder_hvac_in"], \
            para["decoder_hvac_h"], \
            para["decoder_hvac_out"]
        self.De_out_insize, self.De_out_h1, self.De_out_h2, self.De_out_outsize = para["De_out_insize"], \
            para["De_out_h1"], \
            para["De_out_h2"], \
            para["De_out_outsize"]

        # Training hyperpara
        self.lr = para["lr"]
        self.epoch = para["epochs"]
        self.decoLen = para["decoLen"]
        self.encoLen = para["encoLen"]
        self.deco_loss, self.toco_loss = None, None
        # Results
        self.to_outputs, self.en_outputs, self.de_outputs = None, None, None
        self.to_measure, self.en_measure, self.de_measure = None, None, None
        # DDe-norm Results
        self.to_denorm, self.en_denorm, self.de_denorm = None, None, None
        self.to_mea_denorme, self.en_mea_denorm, self.de_mea_denorm = None, None, None

        self.decoder_external = gru_decoder(input_size=self.decoder_external_in,
                                            hidden_size=self.decoder_external_h,
                                            output_size=self.decoder_external_out)
        self.decoder_internal = gru_decoder(input_size=self.decoder_internal_in,
                                            hidden_size=self.decoder_internal_h,
                                            output_size=self.decoder_internal_out)
        self.decoder_hvac = gru_decoder(input_size=self.decoder_hvac_in,
                                        hidden_size=self.decoder_hvac_h,
                                        output_size=self.decoder_hvac_out)
        # Decoder Out
        self.decoder_out = gru_Linear(input_size=self.De_out_insize, h1=self.De_out_h1,
                                      h2=self.De_out_h2, output_size=self.De_out_outsize)

    def train_model(self, dataloder):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize array of losses
        enlosses, delosses, tolosses = np.full(self.epoch, np.nan), \
            np.full(self.epoch, np.nan), \
            np.full(self.epoch, np.nan)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        with trange(self.epoch) as tr:
            for it in tr:
                n_batches = 0
                for input_X, output_y in dataloder:
                    input_X, output_y = input_X.to(device), output_y.to(device)
                    n_batches += 1
                    # Divide input_X to Encoder_X and Decoder_X
                    # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
                    Decoder_y = output_y[:, self.encoLen:, :]
                    # Current update
                    Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
                    # zero the gradient
                    optimizer.zero_grad()
                    # Decoder
                    # External heat gain
                    Decoder_X_Ext = input_X[:, self.encoLen:, [1, 2, 3, 4]]
                    # Internal heat gain
                    Decoder_X_Int = input_X[:, self.encoLen:, [3, 4, 5]]
                    # HVAC
                    Decoder_X_HVAC = input_X[:, self.encoLen:, [7]]

                    # Initial for Decoder Input
                    Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
                    Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
                    Decoder_X_HVAC_ = torch.cat((Current_X_zone, Decoder_X_HVAC[:, [0], :]), 2)

                    # Initial for Decoder Hidden
                    decoder_hidden_Ext_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_external_h).to(device)
                    decoder_hidden_hid_Int_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_internal_h).to(device)
                    decoder_hidden_hid_HVAC_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_hvac_h).to(device)

                    # Decoder Output
                    decoder_output_list = torch.zeros(Decoder_X_Ext.shape[0], Decoder_X_Ext.shape[1], 1).to(device)
                    for i in range(Decoder_X_Ext.shape[1] - 1):
                        decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_,
                                                                                         decoder_hidden_Ext_)
                        decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                             decoder_hidden_hid_Int_)
                        decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                                           decoder_hidden_hid_HVAC_)

                        decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_,
                                                       decoder_output_HVAC_), 2)
                        decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
                        decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)
                        i += 1
                        Decoder_X_Ext_ = torch.cat((decoder_output, Decoder_X_Ext[:, [i], :]), 2)
                        Decoder_X_Int_ = torch.cat((decoder_output, Decoder_X_Int[:, [i], :]), 2)
                        Decoder_X_HVAC_ = torch.cat((decoder_output, Decoder_X_HVAC[:, [i], :]), 2)

                    decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_,
                                                                                     decoder_hidden_Ext_)
                    decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                         decoder_hidden_hid_Int_)
                    decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                                       decoder_hidden_hid_HVAC_)

                    decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
                    decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
                    decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)

                    decoder_loss, encoder_loss, total_loss = 0., 0., 0.
                    deloss = criterion(decoder_output_list, Decoder_y)
                    decoder_loss += deloss.item()
                    # backpropagation
                    deloss.backward()
                    optimizer.step()

                    for p in self.decoder_out.parameters():
                        p.data.clamp_(0)
                    self.decoder_external.gru.weight_hh_l0.data[-12:,:].clamp_(0)
                    self.decoder_external.gru.weight_ih_l0.data[-12:,:].clamp_(0)
                    self.decoder_internal.gru.weight_hh_l0.data[-12:,:].clamp_(0)
                    self.decoder_internal.gru.weight_ih_l0.data[-12:,:].clamp_(0)
                    self.decoder_hvac.gru.weight_hh_l0.data[-12:,:].clamp_(0)
                    self.decoder_hvac.gru.weight_ih_l0.data[-12:,:].clamp_(0)

                encoder_loss /= n_batches
                decoder_loss /= n_batches
                total_loss /= n_batches

                enlosses[it] = encoder_loss
                delosses[it] = decoder_loss
                tolosses[it] = total_loss

                tr.set_postfix(encoder_loss="{0:.6f}".format(encoder_loss), decoder_loss="{0:.6f}".format(decoder_loss),
                               total_loss="{0:.6f}".format(total_loss))

        self.enco_loss, self.deco_loss, self.toco_loss = enlosses, delosses, tolosses

    def test_model(self, dataloder, tempscal):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        to_outputs, en_outputs, de_outputs = [], [], []
        to_measure, en_measure, de_measure = [], [], []

        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            Encoder_y = output_y[:, :self.encoLen, :]
            # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
            # Encoder
            # External heat gain
            Decoder_y = output_y[:, self.encoLen:, :]
            # Current update
            Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
            # Decoder
            # External heat gain
            Decoder_X_Ext = input_X[:, self.encoLen:, [1, 2, 3, 4]]
            # Internal heat gain
            Decoder_X_Int = input_X[:, self.encoLen:, [3, 4, 5]]
            # HVAC
            Decoder_X_HVAC = input_X[:, self.encoLen:, [7]]

            # Initial for Decoder Input
            Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
            Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
            Decoder_X_HVAC_ = torch.cat((Current_X_zone, Decoder_X_HVAC[:, [0], :]), 2)

            # Initial for Decoder Hidden
            decoder_hidden_Ext_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_external_h).to(device)
            decoder_hidden_hid_Int_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_internal_h).to(device)
            decoder_hidden_hid_HVAC_ = torch.zeros(1, Decoder_X_Ext.shape[0], self.decoder_hvac_h).to(device)

            # Decoder Output
            decoder_output_list = torch.zeros(Decoder_X_Ext.shape[0], Decoder_X_Ext.shape[1], 1).to(device)
            for i in range(Decoder_X_Ext.shape[1] - 1):
                decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_,
                                                                                 decoder_hidden_Ext_)
                decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                     decoder_hidden_hid_Int_)
                decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                                   decoder_hidden_hid_HVAC_)

                decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_,
                                               decoder_output_HVAC_), 2)
                decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
                decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)
                i += 1
                Decoder_X_Ext_ = torch.cat((decoder_output, Decoder_X_Ext[:, [i], :]), 2)
                Decoder_X_Int_ = torch.cat((decoder_output, Decoder_X_Int[:, [i], :]), 2)
                Decoder_X_HVAC_ = torch.cat((decoder_output, Decoder_X_HVAC[:, [i], :]), 2)

            decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_,
                                                                             decoder_hidden_Ext_)
            decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                 decoder_hidden_hid_Int_)
            decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                               decoder_hidden_hid_HVAC_)

            decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
            decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
            decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)

            de_outputs.append(decoder_output_list.to("cpu").detach().numpy())
            de_measure.append(Decoder_y.to("cpu").detach().numpy())

        self.de_outputs = de_outputs
        self.de_measure = de_measure

        # De-Norm
        to_out, en_out, de_out, to_mea, en_mea, de_mea = [], [], [], [], [], []
        hidden_ExWall, hidden_Win_S, hidden_Win_C, hidden_Inter, hidden_HVAC, hidden_InWall = [], [], [], [], [], []
        for idx in range(de_outputs[0].shape[0]):
            de_out.append(tempscal.inverse_transform(de_outputs[0][[idx], :, :].reshape(-1, 1)))
            de_mea.append(tempscal.inverse_transform(de_measure[0][[idx], :, :].reshape(-1, 1)))

        self.de_denorm = de_out
        self.de_mea_denorm = de_mea
