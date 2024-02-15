import torch
import torch.nn as nn
import numpy as np
from tqdm import trange

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

class gru_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(gru_encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # define GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bias=True)
        self.Enfc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x_input):
        gru_out, self.hidden = self.gru(x_input)
        output = self.Enfc(gru_out)

        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

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

    def forward(self, x_input, encoder_hidden_states):
        gru_out, self.hidden = self.gru(x_input, encoder_hidden_states)
        output = self.Defc(gru_out)

        return output, self.hidden

class gru_seq2seq(nn.Module):

    def __init__(self, para):

        super(gru_seq2seq, self).__init__()
        # Parameters
        # Encoder
        self.encoder_external_in, self.encoder_external_h, self.encoder_external_out = para["encoder_external_in"], \
            para["encoder_external_h"], \
            para["encoder_external_out"]
        self.encoder_internal_in, self.encoder_internal_h, self.encoder_internal_out = para["encoder_internal_in"], \
            para["encoder_internal_h"], \
            para["encoder_internal_out"]
        self.encoder_hvac_in, self.encoder_hvac_h, self.encoder_hvac_out = para["encoder_hvac_in"], \
            para["encoder_hvac_h"], \
            para["encoder_hvac_out"]

        # Decoder
        self.decoder_external_in, self.decoder_external_h, self.decoder_external_out = para["decoder_external_in"], \
            para["decoder_external_h"], \
            para["decoder_external_out"]
        self.decoder_internal_in, self.decoder_internal_h, self.decoder_internal_out = para["decoder_internal_in"], \
            para["decoder_internal_h"], \
            para["decoder_internal_out"]
        self.decoder_hvac_in, self.decoder_hvac_h, self.decoder_hvac_out = para["decoder_hvac_in"], \
            para["decoder_hvac_h"], \
            para["decoder_hvac_out"]

        # FC Layer
        self.En_out_insize, self.En_out_h1, self.En_out_h2, self.En_out_outsize = para["En_out_insize"], \
            para["En_out_h1"], \
            para["En_out_h2"], \
            para["En_out_outsize"]
        self.De_out_insize, self.De_out_h1, self.De_out_h2, self.De_out_outsize = para["De_out_insize"], \
            para["De_out_h1"], \
            para["De_out_h2"], \
            para["De_out_outsize"]

        # Training hyperpara
        self.lr = para["lr"]
        self.epoch = para["epochs"]
        self.encoLen = para["encoLen"]
        self.decoLen = para["decoLen"]
        self.enco_loss, self.deco_loss, self.toco_loss = None, None, None
        # Results
        self.to_outputs, self.en_outputs, self.de_outputs = None, None, None
        self.to_measure, self.en_measure, self.de_measure = None, None, None
        # DDe-norm Results
        self.to_denorm, self.en_denorm, self.de_denorm = None, None, None
        self.to_mea_denorme, self.en_mea_denorm, self.de_mea_denorm = None, None, None

        # Model
        # Encoder
        self.encoder_external = gru_encoder(input_size=self.encoder_external_in,
                                            hidden_size=self.encoder_external_h,
                                            output_size=self.encoder_external_out)
        self.encoder_internal = gru_encoder(input_size=self.encoder_internal_in,
                                            hidden_size=self.encoder_internal_h,
                                            output_size=self.encoder_internal_out)
        self.encoder_hvac = gru_encoder(input_size=self.encoder_hvac_in,
                                        hidden_size=self.encoder_hvac_h,
                                        output_size=self.encoder_hvac_out)

        # Decoder
        self.decoder_external = gru_decoder(input_size=self.decoder_external_in,
                                            hidden_size=self.decoder_external_h,
                                            output_size=self.decoder_external_out)
        self.decoder_internal = gru_decoder(input_size=self.decoder_internal_in,
                                            hidden_size=self.decoder_internal_h,
                                            output_size=self.decoder_internal_out)
        self.decoder_hvac = gru_decoder(input_size=self.decoder_hvac_in,
                                        hidden_size=self.decoder_hvac_h,
                                        output_size=self.decoder_hvac_out)

        # Encoder Out
        self.encoder_out = gru_Linear(input_size=self.En_out_insize, h1=self.En_out_h1,
                                      h2=self.En_out_h2, output_size=self.En_out_outsize)
        # Decoder Out
        self.decoder_out = gru_Linear(input_size=self.De_out_insize, h1=self.De_out_h1,
                                      h2=self.De_out_h2, output_size=self.De_out_outsize)

    def train_model(self, dataloder):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                    Encoder_y = output_y[:, :self.encoLen, :]
                    Decoder_y = output_y[:, self.encoLen:, :]
                    # Current update
                    Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
                    # zero the gradient
                    optimizer.zero_grad()
                    # Encoder
                    # External heat gain
                    Encoder_X_Ext = input_X[:, :self.encoLen, [0, 1, 2, 3, 4]]
                    # Internal heat gain
                    Encoder_X_Int = input_X[:, :self.encoLen, [0, 3, 4, 5]]
                    # HVAC
                    Encoder_X_HVAC = input_X[:, :self.encoLen, [0, 7]]
                    # Decoder
                    # External heat gain
                    Decoder_X_Ext = input_X[:, self.encoLen:, [1, 2, 3, 4]]
                    # Internal heat gain
                    Decoder_X_Int = input_X[:, self.encoLen:, [3, 4, 5]]
                    # HVAC
                    Decoder_X_HVAC = input_X[:, self.encoLen:, [7]]

                    # Calculate
                    encoder_out_Ext, encoder_hid_Ext = self.encoder_external(Encoder_X_Ext)
                    encoder_out_Int, encoder_hid_Int = self.encoder_internal(Encoder_X_Int)
                    encoder_out_HVAC, encoder_hid_HVAC = self.encoder_hvac(Encoder_X_HVAC)

                    # embedding
                    encoder_out_embed = torch.cat((encoder_out_Ext, encoder_out_Int, encoder_out_HVAC), 2)
                    encoder_output = self.encoder_out(encoder_out_embed)

                    # Initial for Decoder Input
                    Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
                    Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
                    Decoder_X_HVAC_ = torch.cat((Current_X_zone, Decoder_X_HVAC[:, [0], :]), 2)

                    # Initial for Decoder Hidden
                    decoder_hidden_Ext_ = encoder_hid_Ext
                    decoder_hidden_hid_Int_ = encoder_hid_Int
                    decoder_hidden_hid_HVAC_ = encoder_hid_HVAC

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
                    outputs = torch.cat((encoder_output, decoder_output_list), 1)
                    deloss = criterion(encoder_output, Encoder_y)
                    enloss = criterion(decoder_output_list, Decoder_y)
                    toloss = criterion(outputs, output_y)

                    decoder_loss += deloss.item()
                    encoder_loss += enloss.item()
                    total_loss += toloss.item()

                    # backpropagation
                    toloss.backward()
                    optimizer.step()

                    for p in self.decoder_out.parameters():
                        p.data.clamp_(0)
                    for p in self.encoder_out.parameters():
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
            Decoder_y = output_y[:, self.encoLen:, :]
            # Current update
            Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
            # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
            # Encoder
            # External heat gain
            Encoder_X_Ext = input_X[:, :self.encoLen, [0, 1, 2, 3, 4]]
            # Internal heat gain
            Encoder_X_Int = input_X[:, :self.encoLen, [0, 3, 4, 5]]
            # HVAC
            Encoder_X_HVAC = input_X[:, :self.encoLen, [0, 7]]

            # Decoder
            # External heat gain
            Decoder_X_Ext = input_X[:, self.encoLen:, [1, 2, 3, 4]]
            # Internal heat gain
            Decoder_X_Int = input_X[:, self.encoLen:, [3, 4, 5]]
            # HVAC
            Decoder_X_HVAC = input_X[:, self.encoLen:, [7]]

            # Calculate
            encoder_out_Ext, encoder_hid_Ext = self.encoder_external(Encoder_X_Ext)
            encoder_out_Int, encoder_hid_Int = self.encoder_internal(Encoder_X_Int)
            encoder_out_HVAC, encoder_hid_HVAC = self.encoder_hvac(Encoder_X_HVAC)
            # embedding
            encoder_out_embed = torch.cat((encoder_out_Ext, encoder_out_Int, encoder_out_HVAC), 2)
            encoder_output = self.encoder_out(encoder_out_embed)

            # Initial for Decoder Input
            Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
            Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
            Decoder_X_HVAC_ = torch.cat((Current_X_zone, Decoder_X_HVAC[:, [0], :]), 2)

            # Initial for Decoder Hidden
            decoder_hidden_Ext_ = encoder_hid_Ext
            decoder_hidden_hid_Int_ = encoder_hid_Int
            decoder_hidden_hid_HVAC_ = encoder_hid_HVAC

            # Decoder Output
            decoder_output_list = torch.zeros(Decoder_X_Ext.shape[0], Decoder_X_Ext.shape[1], 1).to(device)
            for i in range(Decoder_X_Ext.shape[1] - 1):
                decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
                decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                     decoder_hidden_hid_Int_)
                decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                                   decoder_hidden_hid_HVAC_)

                decoder_out_embed = torch.cat(
                    (decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
                decoder_output = self.decoder_out(decoder_out_embed)
                decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)
                i += 1
                Decoder_X_Ext_ = torch.cat((decoder_output, Decoder_X_Ext[:, [i], :]), 2)
                Decoder_X_Int_ = torch.cat((decoder_output, Decoder_X_Int[:, [i], :]), 2)
                Decoder_X_HVAC_ = torch.cat((decoder_output, Decoder_X_HVAC[:, [i], :]), 2)

            decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
            decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                 decoder_hidden_hid_Int_)
            decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                               decoder_hidden_hid_HVAC_)

            decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_,
                                           decoder_output_HVAC_), 2)
            decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
            decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)

            outputs = torch.cat((encoder_output, decoder_output_list), 1)
            to_outputs.append(outputs.to("cpu").detach().numpy())
            en_outputs.append(encoder_output.to("cpu").detach().numpy())
            de_outputs.append(decoder_output_list.to("cpu").detach().numpy())
            to_measure.append(output_y.to("cpu").detach().numpy())
            en_measure.append(Encoder_y.to("cpu").detach().numpy())
            de_measure.append(Decoder_y.to("cpu").detach().numpy())

        self.to_outputs, self.en_outputs, self.de_outputs = to_outputs, en_outputs, de_outputs
        self.to_measure, self.en_measure, self.de_measure = to_measure, en_measure, de_measure

        # De-Norm
        to_out, en_out, de_out, to_mea, en_mea, de_mea = [], [], [], [], [], []
        hidden_ExWall, hidden_Win_S, hidden_Win_C, hidden_Inter, hidden_HVAC, hidden_InWall = [], [], [], [], [], []
        for idx in range(to_outputs[0].shape[0]):
            to_out.append(tempscal.inverse_transform(to_outputs[0][[idx], :, :].reshape(-1, 1)))
            en_out.append(tempscal.inverse_transform(en_outputs[0][[idx], :, :].reshape(-1, 1)))
            de_out.append(tempscal.inverse_transform(de_outputs[0][[idx], :, :].reshape(-1, 1)))
            to_mea.append(tempscal.inverse_transform(to_measure[0][[idx], :, :].reshape(-1, 1)))
            en_mea.append(tempscal.inverse_transform(en_measure[0][[idx], :, :].reshape(-1, 1)))
            de_mea.append(tempscal.inverse_transform(de_measure[0][[idx], :, :].reshape(-1, 1)))

        self.to_denorm = to_out
        self.en_denorm = en_out
        self.de_denorm = de_out
        self.to_mea_denorme = to_mea
        self.en_mea_denorm = en_mea
        self.de_mea_denorm = de_mea
    def check_model(self, dataloder, tempscal, check):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        to_outputs, en_outputs, de_outputs = [], [], []
        to_measure, en_measure, de_measure = [], [], []

        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            Encoder_y = output_y[:, :self.encoLen, :]
            Decoder_y = output_y[:, self.encoLen:, :]
            # Current update
            Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
            # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
            # Encoder
            # External heat gain
            Encoder_X_Ext = input_X[:, :self.encoLen, [0, 1, 2, 3, 4]]
            # Internal heat gain
            Encoder_X_Int = input_X[:, :self.encoLen, [0, 3, 4, 5]]
            # HVAC
            Encoder_X_HVAC = input_X[:, :self.encoLen, [0, 7]]

            # Decoder
            # External heat gain
            Decoder_X_Ext = input_X[:, self.encoLen:, [1, 2, 3, 4]]
            if check=='Tempmin':
                Decoder_X_Ext[:,:,[0]] = torch.ones_like(Decoder_X_Ext[:,:,[0]])*0
            if check=='Tempmax':
                Decoder_X_Ext[:,:,[0]] = torch.ones_like(Decoder_X_Ext[:,:,[0]])*1
            if check=='Solmin':
                Decoder_X_Ext[:,:,[1]] = torch.ones_like(Decoder_X_Ext[:,:,[1]])*0
            if check=='Solmax':
                Decoder_X_Ext[:,:,[1]] = torch.ones_like(Decoder_X_Ext[:,:,[1]])*1
            # Internal heat gain
            Decoder_X_Int = input_X[:, self.encoLen:, [3, 4, 5]]
            # HVAC
            Decoder_X_HVAC = input_X[:, self.encoLen:, [7]]
            if check=='HVACmin':
                Decoder_X_HVAC = torch.ones_like(Decoder_X_HVAC)*-1
            if check=='HVACmax':
                Decoder_X_HVAC = torch.ones_like(Decoder_X_HVAC)*0

            # Calculate
            encoder_out_Ext, encoder_hid_Ext = self.encoder_external(Encoder_X_Ext)
            encoder_out_Int, encoder_hid_Int = self.encoder_internal(Encoder_X_Int)
            encoder_out_HVAC, encoder_hid_HVAC = self.encoder_hvac(Encoder_X_HVAC)
            # embedding
            encoder_out_embed = torch.cat((encoder_out_Ext, encoder_out_Int, encoder_out_HVAC), 2)
            encoder_output = self.encoder_out(encoder_out_embed)

            # Initial for Decoder Input
            Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
            Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
            Decoder_X_HVAC_ = torch.cat((Current_X_zone, Decoder_X_HVAC[:, [0], :]), 2)

            # Initial for Decoder Hidden
            decoder_hidden_Ext_ = encoder_hid_Ext
            decoder_hidden_hid_Int_ = encoder_hid_Int
            decoder_hidden_hid_HVAC_ = encoder_hid_HVAC

            # Decoder Output
            decoder_output_list = torch.zeros(Decoder_X_Ext.shape[0], Decoder_X_Ext.shape[1], 1).to(device)
            for i in range(Decoder_X_Ext.shape[1] - 1):
                decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
                decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                     decoder_hidden_hid_Int_)
                decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                                   decoder_hidden_hid_HVAC_)

                decoder_out_embed = torch.cat(
                    (decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
                decoder_output = self.decoder_out(decoder_out_embed)
                decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)
                i += 1
                Decoder_X_Ext_ = torch.cat((decoder_output, Decoder_X_Ext[:, [i], :]), 2)
                Decoder_X_Int_ = torch.cat((decoder_output, Decoder_X_Int[:, [i], :]), 2)
                Decoder_X_HVAC_ = torch.cat((decoder_output, Decoder_X_HVAC[:, [i], :]), 2)

            decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
            decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_,
                                                                                 decoder_hidden_hid_Int_)
            decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_,
                                                                               decoder_hidden_hid_HVAC_)

            decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_,
                                           decoder_output_HVAC_), 2)
            decoder_output = self.decoder_out(decoder_out_embed)  # Estimated Tzone for next step
            decoder_output_list[:, i, :] = torch.squeeze(decoder_output, dim=1)

            outputs = torch.cat((encoder_output, decoder_output_list), 1)
            to_outputs.append(outputs.to("cpu").detach().numpy())
            en_outputs.append(encoder_output.to("cpu").detach().numpy())
            de_outputs.append(decoder_output_list.to("cpu").detach().numpy())
            to_measure.append(output_y.to("cpu").detach().numpy())
            en_measure.append(Encoder_y.to("cpu").detach().numpy())
            de_measure.append(Decoder_y.to("cpu").detach().numpy())

        self.to_outputs, self.en_outputs, self.de_outputs = to_outputs, en_outputs, de_outputs
        self.to_measure, self.en_measure, self.de_measure = to_measure, en_measure, de_measure

        # De-Norm
        to_out, en_out, de_out, to_mea, en_mea, de_mea = [], [], [], [], [], []
        hidden_ExWall, hidden_Win_S, hidden_Win_C, hidden_Inter, hidden_HVAC, hidden_InWall = [], [], [], [], [], []
        for idx in range(to_outputs[0].shape[0]):
            to_out.append(tempscal.inverse_transform(to_outputs[0][[idx], :, :].reshape(-1, 1)))
            en_out.append(tempscal.inverse_transform(en_outputs[0][[idx], :, :].reshape(-1, 1)))
            de_out.append(tempscal.inverse_transform(de_outputs[0][[idx], :, :].reshape(-1, 1)))
            to_mea.append(tempscal.inverse_transform(to_measure[0][[idx], :, :].reshape(-1, 1)))
            en_mea.append(tempscal.inverse_transform(en_measure[0][[idx], :, :].reshape(-1, 1)))
            de_mea.append(tempscal.inverse_transform(de_measure[0][[idx], :, :].reshape(-1, 1)))

        self.to_denorm = to_out
        self.en_denorm = en_out
        self.de_denorm = de_out
        self.to_mea_denorme = to_mea
        self.en_mea_denorm = en_mea
        self.de_mea_denorm = de_mea


