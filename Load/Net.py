import torch
import torch.nn as nn
import numpy as np
from tqdm import trange


class gru_prepare(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        super(gru_prepare, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bias=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x_input):

        gru_out, self.hidden = self.gru(x_input)
        output = self.fc(gru_out)

        return output, self.hidden

    def init_hidden(self, batch_size):

        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

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
        #Embedding_state = self.relu(Embedding_state)

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

class gnn_connect(nn.Module):

    def __init__(self, input_size, h1, output_size, adjacency):
        super(gnn_connect, self).__init__()
        self.Fc1 = nn.Linear(in_features=input_size, out_features=h1, bias=True)
        self.Fc2 = nn.Linear(in_features=h1, out_features=output_size, bias=True)
        self.adjacency = adjacency
        self.relu = nn.ReLU()

    def forward(self, x_mix):
        Embedding_state = torch.einsum('ijk,kl->ijl', x_mix, self.adjacency)
        Embedding_state = self.Fc1(Embedding_state)
        Embedding_state = self.relu(Embedding_state)
        Embedding_state = self.Fc2(Embedding_state)

        return Embedding_state

class gru_seq2seq(nn.Module):

    def __init__(self):
        super(gru_seq2seq, self).__init__()
        self.para = None


    def initial(self, para):
        self.lr = para["lr"]
        self.epoch = para["epochs"]
        self.encoLen = para["encoLen"]
        self.decoLen = para["decoLen"]
        if para["Task"] == "Temperature_Prediction":
            # Parameters
            # Encoder
            self.encoder_external_in, self.encoder_external_h, self.encoder_external_out = para["encoder_external_in"], para["encoder_external_h"], para["encoder_external_out"]
            self.encoder_internal_in, self.encoder_internal_h, self.encoder_internal_out = para["encoder_internal_in"], para["encoder_internal_h"], para["encoder_internal_out"]
            self.encoder_hvac_in, self.encoder_hvac_h, self.encoder_hvac_out = para["encoder_hvac_in"], para["encoder_hvac_h"], para["encoder_hvac_out"]
            self.encoder_other_in, self.encoder_other_h, self.encoder_other_out = para["encoder_other_in"], para["encoder_other_h"], para["encoder_other_out"]
            # Decoder
            self.decoder_external_in, self.decoder_external_h, self.decoder_external_out = para["decoder_external_in"], para["decoder_external_h"], para["decoder_external_out"]
            self.decoder_internal_in, self.decoder_internal_h, self.decoder_internal_out = para["decoder_internal_in"], para["decoder_internal_h"], para["decoder_internal_out"]
            self.decoder_hvac_in, self.decoder_hvac_h, self.decoder_hvac_out = para["decoder_hvac_in"], para["decoder_hvac_h"], para["decoder_hvac_out"]
            self.decoder_other_in, self.decoder_other_h, self.decoder_other_out = para["decoder_other_in"], para["decoder_other_h"], para["decoder_other_out"]
            # FC Layer
            self.En_out_insize, self.En_out_h1, self.En_out_h2, self.En_out_outsize = para["En_out_insize"], para["En_out_h1"], para["En_out_h2"], para["En_out_outsize"]
            self.De_out_insize, self.De_out_h1, self.De_out_h2, self.De_out_outsize = para["De_out_insize"], para["De_out_h1"], para["De_out_h2"], para["De_out_outsize"]
            # GNN Layer
            self.enco_gnn_in, self.enco_gnn_h, self.enco_gnn_out = para["enco_gnn_in"], para["enco_gnn_h"], para["enco_gnn_out"]
            self.deco_gnn_in, self.deco_gnn_h, self.deco_gnn_out = para["deco_gnn_in"], para["deco_gnn_h"], para["deco_gnn_out"]
            self.adjMatrix = para["adjMatrix"]

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
            self.encoder_external = gru_encoder(input_size=self.encoder_external_in, hidden_size=self.encoder_external_h,
                                              output_size=self.encoder_external_out)
            self.encoder_internal = gru_encoder(input_size=self.encoder_internal_in, hidden_size=self.encoder_internal_h,
                                              output_size= self.encoder_internal_out)
            self.encoder_hvac = gru_encoder(input_size=self.encoder_hvac_in, hidden_size=self.encoder_hvac_h,
                                              output_size=self.encoder_hvac_out)
            self.encoder_other = gru_encoder(input_size=self.encoder_other_in, hidden_size=self.encoder_other_h,
                                              output_size=self.encoder_other_out)

            # Decoder
            self.decoder_external = gru_decoder(input_size=self.decoder_external_in, hidden_size=self.decoder_external_h,
                                              output_size=self.decoder_external_out)
            self.decoder_internal = gru_decoder(input_size=self.decoder_internal_in, hidden_size=self.decoder_internal_h,
                                              output_size= self.decoder_internal_out)
            self.decoder_hvac = gru_decoder(input_size=self.decoder_hvac_in, hidden_size=self.decoder_hvac_h,
                                              output_size=self.decoder_hvac_out)
            self.decoder_other = gru_decoder(input_size=self.decoder_other_in, hidden_size=self.decoder_other_h,
                                              output_size=self.decoder_other_out)
            #GNN Agg
            self.enco_gnn = gnn_connect(input_size=self.enco_gnn_in, h1=self.enco_gnn_h, output_size=self.enco_gnn_out, adjacency=self.adjMatrix.to('cuda'))
            self.deco_gnn = gnn_connect(input_size=self.deco_gnn_in, h1=self.deco_gnn_h, output_size=self.deco_gnn_out, adjacency=self.adjMatrix.to('cuda'))

            #Encoder Out
            self.encoder_out = gru_Linear(input_size=self.En_out_insize, h1=self.En_out_h1,
                                          h2=self.En_out_h2, output_size=self.En_out_outsize)
            #Decoder Out
            self.decoder_out = gru_Linear(input_size=self.De_out_insize, h1=self.De_out_h1,
                                          h2=self.De_out_h2, output_size=self.De_out_outsize)
        if para["Task"] == "loadpred":
            # HVAC Air Load Layer
            self.encoder_pHVAC_in, self.encoder_pHVAC_h, self.encoder_pHVAC_out = para["encoder_pHVAC_in"], para["encoder_pHVAC_h"], para["encoder_pHVAC_out"]
            self.decoder_pHVAC_in, self.decoder_pHVAC_h, self.decoder_pHVAC_out = para["decoder_pHVAC_in"], para["decoder_pHVAC_h"], para["decoder_pHVAC_out"]
            # Other Load Layer
            self.encoder_other_in, self.encoder_other_h, self.encoder_other_out = para["encoder_other_in"], para["encoder_other_h"], para["encoder_other_out"]
            self.decoder_other_in, self.decoder_other_h, self.decoder_other_out = para["decoder_other_in"], para["decoder_other_h"], para["decoder_other_out"]
            # HVAC Load Layer
            self.Phvac_in, self.Phvac_h1, self.Phvac_h2, self.Phvac_out = para["Phvac_in"], para["Phvac_h1"], para["Phvac_h2"], para["Phvac_out"]
            # HVAC Electric Layer
            self.Ehvac_in, self.Ehvac_h1, self.Ehvac_h2, self.Ehvac_out = para["Ehvac_in"], para["Ehvac_h1"], para["Ehvac_h2"], para["Ehvac_out"]
            # GNN Layer
            self.enco_gnn_in, self.enco_gnn_h, self.enco_gnn_out = para["enco_gnn_in"], para["enco_gnn_h"], para["enco_gnn_out"]
            self.deco_gnn_in, self.deco_gnn_h, self.deco_gnn_out = para["deco_gnn_in"], para["deco_gnn_h"], para["deco_gnn_out"]
            self.adjMatrix = para["adjMatrix"]


            # Encoder
            self.encoder_pHVAC = gru_encoder(input_size=self.encoder_pHVAC_in,
                                             hidden_size=self.encoder_pHVAC_h,
                                             output_size=self.encoder_pHVAC_out)
            self.encoder_other = gru_encoder(input_size=self.encoder_other_in,
                                             hidden_size=self.encoder_other_h,
                                             output_size=self.encoder_other_out)
            self.encoder_Ehvac = gru_Linear(input_size=self.Ehvac_in, h1=self.Ehvac_h1, h2=self.Ehvac_h2, output_size=self.Ehvac_out)
            self.encoder_Phvac = gru_Linear(input_size=self.Phvac_in, h1=self.Phvac_h1, h2=self.Phvac_h2, output_size=self.Phvac_out)
            # Decoder
            self.decoder_pHVAC = gru_decoder(input_size=self.decoder_pHVAC_in,
                                             hidden_size=self.decoder_pHVAC_h,
                                             output_size=self.decoder_pHVAC_out)
            self.decoder_other = gru_decoder(input_size=self.decoder_other_in,
                                             hidden_size=self.decoder_other_h,
                                             output_size=self.decoder_other_out)
            self.decoder_Ehvac = gru_Linear(input_size=self.Ehvac_in, h1=self.Ehvac_h1, h2=self.Ehvac_h2, output_size=self.Ehvac_out)
            self.decoder_Phvac = gru_Linear(input_size=self.Phvac_in, h1=self.Phvac_h1, h2=self.Phvac_h2, output_size=self.Phvac_out)
            # GNN
            self.enco_gnn = gnn_connect(input_size=self.enco_gnn_in, h1=self.enco_gnn_h, output_size=self.enco_gnn_out,
                                        adjacency=self.adjMatrix.to('cuda'))
            self.deco_gnn = gnn_connect(input_size=self.deco_gnn_in, h1=self.deco_gnn_h, output_size=self.deco_gnn_out,
                                        adjacency=self.adjMatrix.to('cuda'))


    def train_load_estimation_model(self, dataloder, zone_index):
        self.zone_index = zone_index
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize array of losses
        enlosses, delosses, tolosses = np.full(self.epoch, np.nan), np.full(self.epoch, np.nan), np.full(self.epoch, np.nan)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        with trange(self.epoch) as tr:
            for it in tr:
                n_batches = 0
                for input_X, output_y in dataloder:
                    input_X, output_y = input_X.to(device), output_y.to(device)
                    n_batches += 1
                    # Divide input_X to Encoder_X and Decoder_X
                    # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac; 8,9,10,11,12Adj; Ehvac, Total
                    Encoder_y = output_y[:, :self.encoLen, [7]]
                    Decoder_y = output_y[:, self.encoLen:, [7]]
                    # zero the gradient
                    optimizer.zero_grad()
                    # Encoder
                    Encoder_X_Phvac = input_X[:, :self.encoLen, [1,2,3,4,5,6]]
                    # Decoder
                    Decoder_X_Phvac = input_X[:, self.encoLen:, [1,2,3,4,5,6]]

                    encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
                    decoder_hidden_Phvac = encoder_hid_Phvac
                    decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
                    encoder_out_Phvac = self.encoder_Phvac(encoder_out_Phvac)
                    decoder_out_Phvac = self.decoder_Phvac(decoder_out_Phvac)

                    decoder_loss, encoder_loss, total_loss = 0., 0., 0.
                    outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
                    deloss = criterion(encoder_out_Phvac, Encoder_y)
                    enloss = criterion(decoder_out_Phvac, Decoder_y)
                    toloss = criterion(outputs, output_y[:, :, [7]])

                    decoder_loss += deloss.item()
                    encoder_loss += enloss.item()
                    total_loss += toloss.item()

                    # backpropagation
                    toloss.backward()
                    optimizer.step()

                encoder_loss /= n_batches
                decoder_loss /= n_batches
                total_loss /= n_batches

                enlosses[it] = encoder_loss
                delosses[it] = decoder_loss
                tolosses[it] = total_loss

                tr.set_postfix(encoder_loss="{0:.6f}".format(encoder_loss), decoder_loss="{0:.6f}".format(decoder_loss),
                               total_loss="{0:.6f}".format(total_loss))

    def multi_train_load_estimation_model(self, dataloder, zone_index):
        self.zone_index = zone_index
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize array of losses
        enlosses, delosses, tolosses = np.full(self.epoch, np.nan), np.full(self.epoch, np.nan), np.full(self.epoch, np.nan)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        with trange(self.epoch) as tr:
            for it in tr:
                n_batches = 0
                for input_X, output_y in dataloder:
                    input_X, output_y = input_X.to(device), output_y.to(device)
                    n_batches += 1
                    # Divide input_X to Encoder_X and Decoder_X
                    # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac; 8,9,10,11,12Adj; Ehvac, Total
                    Encoder_y = output_y[:, :self.encoLen, [7]]
                    Decoder_y = output_y[:, self.encoLen:, [7]]
                    # zero the gradient
                    optimizer.zero_grad()
                    # Encoder
                    Encoder_X_Adj = self.enco_gnn(input_X[:, :self.encoLen, [8, 9, 10, 11, 12]])
                    Encoder_X_Phvac = input_X[:, :self.encoLen, [1,2,3,4,5,6]]
                    Encoder_X_Phvac = torch.cat((Encoder_X_Phvac, Encoder_X_Adj), 2)
                    # Decoder
                    Decoder_X_Adj = self.deco_gnn(input_X[:, self.encoLen:, [8, 9, 10, 11, 12]])
                    Decoder_X_Phvac = input_X[:, self.encoLen:, [1,2,3,4,5,6]]
                    Decoder_X_Phvac = torch.cat((Decoder_X_Phvac, Decoder_X_Adj), 2)

                    encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
                    decoder_hidden_Phvac = encoder_hid_Phvac
                    decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
                    encoder_out_Phvac = self.encoder_Phvac(encoder_out_Phvac)
                    decoder_out_Phvac = self.decoder_Phvac(decoder_out_Phvac)

                    decoder_loss, encoder_loss, total_loss = 0., 0., 0.
                    outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
                    deloss = criterion(encoder_out_Phvac, Encoder_y)
                    enloss = criterion(decoder_out_Phvac, Decoder_y)
                    toloss = criterion(outputs, output_y[:, :, [7]])

                    decoder_loss += deloss.item()
                    encoder_loss += enloss.item()
                    total_loss += toloss.item()

                    # backpropagation
                    toloss.backward()
                    optimizer.step()

                encoder_loss /= n_batches
                decoder_loss /= n_batches
                total_loss /= n_batches

                enlosses[it] = encoder_loss
                delosses[it] = decoder_loss
                tolosses[it] = total_loss

                tr.set_postfix(encoder_loss="{0:.6f}".format(encoder_loss), decoder_loss="{0:.6f}".format(decoder_loss),
                               total_loss="{0:.6f}".format(total_loss))

    def test_load_estimation_model(self, dataloder, loadscal):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        to_outputs, en_outputs, de_outputs = [], [], []

        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            # Encoder
            Encoder_X_Phvac = input_X[:, :self.encoLen, [1,2,3,4,5,6]]
            # Decoder
            Decoder_X_Phvac = input_X[:, self.encoLen:, [1,2,3,4,5,6]]

            encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
            decoder_hidden_Phvac = encoder_hid_Phvac
            decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
            encoder_out_Phvac = self.encoder_Phvac(encoder_out_Phvac)
            decoder_out_Phvac = self.decoder_Phvac(decoder_out_Phvac)

            outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
            to_outputs.append(outputs.to("cpu").detach().numpy())
            en_outputs.append(encoder_out_Phvac.to("cpu").detach().numpy())
            de_outputs.append(decoder_out_Phvac.to("cpu").detach().numpy())

        self.to_outputs, self.en_outputs, self.de_outputs = to_outputs, en_outputs, de_outputs
        # De-Norm
        to_out, en_out, de_out = [], [], []
        for idx in range(to_outputs[0].shape[0]):
            to_out.append(loadscal.inverse_transform(to_outputs[0][[idx],:,:].reshape(-1, 1)))
            en_out.append(loadscal.inverse_transform(en_outputs[0][[idx],:,:].reshape(-1, 1)))
            de_out.append(loadscal.inverse_transform(de_outputs[0][[idx],:,:].reshape(-1, 1)))

        self.to_denorm = to_out
        self.en_denorm = en_out
        self.de_denorm = de_out

    def multi_test_load_estimation_model(self, dataloder, loadscal):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        to_outputs, en_outputs, de_outputs = [], [], []

        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            # Encoder
            Encoder_X_Adj = self.enco_gnn(input_X[:, :self.encoLen, [8, 9, 10, 11, 12]])
            Encoder_X_Phvac = input_X[:, :self.encoLen, [1, 2, 3, 4, 5, 6]]
            Encoder_X_Phvac = torch.cat((Encoder_X_Phvac, Encoder_X_Adj), 2)
            # Decoder
            Decoder_X_Adj = self.deco_gnn(input_X[:, self.encoLen:, [8, 9, 10, 11, 12]])
            Decoder_X_Phvac = input_X[:, self.encoLen:, [1, 2, 3, 4, 5, 6]]
            Decoder_X_Phvac = torch.cat((Decoder_X_Phvac, Decoder_X_Adj), 2)

            encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
            decoder_hidden_Phvac = encoder_hid_Phvac
            decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
            encoder_out_Phvac = self.encoder_Phvac(encoder_out_Phvac)
            decoder_out_Phvac = self.decoder_Phvac(decoder_out_Phvac)

            outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
            to_outputs.append(outputs.to("cpu").detach().numpy())
            en_outputs.append(encoder_out_Phvac.to("cpu").detach().numpy())
            de_outputs.append(decoder_out_Phvac.to("cpu").detach().numpy())

        self.to_outputs, self.en_outputs, self.de_outputs = to_outputs, en_outputs, de_outputs
        # De-Norm
        to_out, en_out, de_out = [], [], []
        for idx in range(to_outputs[0].shape[0]):
            to_out.append(loadscal.inverse_transform(to_outputs[0][[idx],:,:].reshape(-1, 1)))
            en_out.append(loadscal.inverse_transform(en_outputs[0][[idx],:,:].reshape(-1, 1)))
            de_out.append(loadscal.inverse_transform(de_outputs[0][[idx],:,:].reshape(-1, 1)))

        self.to_denorm = to_out
        self.en_denorm = en_out
        self.de_denorm = de_out

    def train_energy_estimation_model(self, dataloder, zone_index, y_index):
        self.zone_index = zone_index
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize array of losses
        enlosses, delosses, tolosses = np.full(self.epoch, np.nan), np.full(self.epoch, np.nan), np.full(self.epoch, np.nan)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        with trange(self.epoch) as tr:
            for it in tr:
                n_batches = 0
                for input_X, output_y in dataloder:
                    input_X, output_y = input_X.to(device), output_y.to(device)
                    n_batches += 1
                    # Divide input_X to Encoder_X and Decoder_X
                    # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac; 8,9,10,11,12Adj.
                    Encoder_y = output_y[:, :self.encoLen, [y_index]]
                    Decoder_y = output_y[:, self.encoLen:, [y_index]]
                    # zero the gradient
                    optimizer.zero_grad()
                    # Encoder
                    Encoder_X_Phvac = input_X[:, :self.encoLen, [1,2,3,4,5,6]]
                    # Decoder
                    Decoder_X_Phvac = input_X[:, self.encoLen:, [1,2,3,4,5,6]]

                    encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
                    decoder_hidden_Phvac = encoder_hid_Phvac
                    decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
                    encoder_out_Phvac = self.encoder_Ehvac(encoder_out_Phvac)
                    decoder_out_Phvac = self.decoder_Ehvac(decoder_out_Phvac)

                    decoder_loss, encoder_loss, total_loss = 0., 0., 0.
                    outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
                    deloss = criterion(encoder_out_Phvac, Encoder_y)
                    enloss = criterion(decoder_out_Phvac, Decoder_y)
                    toloss = criterion(outputs, output_y[:, :, [y_index]])

                    decoder_loss += deloss.item()
                    encoder_loss += enloss.item()
                    total_loss += toloss.item()

                    # backpropagation
                    toloss.backward()
                    optimizer.step()

                encoder_loss /= n_batches
                decoder_loss /= n_batches
                total_loss /= n_batches

                enlosses[it] = encoder_loss
                delosses[it] = decoder_loss
                tolosses[it] = total_loss

                tr.set_postfix(encoder_loss="{0:.6f}".format(encoder_loss), decoder_loss="{0:.6f}".format(decoder_loss),
                               total_loss="{0:.6f}".format(total_loss))

    def test_energy_estimation_model(self, dataloder, loadscal):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        to_outputs, en_outputs, de_outputs = [], [], []

        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            # Encoder
            Encoder_X_Phvac = input_X[:, :self.encoLen, [1,2,3,4,5,6]]
            # Decoder
            Decoder_X_Phvac = input_X[:, self.encoLen:, [1,2,3,4,5,6]]

            encoder_out_Phvac, encoder_hid_Phvac = self.encoder_pHVAC(Encoder_X_Phvac)
            decoder_hidden_Phvac = encoder_hid_Phvac
            decoder_out_Phvac, _ = self.decoder_pHVAC(Decoder_X_Phvac, decoder_hidden_Phvac)
            encoder_out_Phvac = self.encoder_Ehvac(encoder_out_Phvac)
            decoder_out_Phvac = self.decoder_Ehvac(decoder_out_Phvac)

            outputs = torch.cat((encoder_out_Phvac, decoder_out_Phvac), 1)
            to_outputs.append(outputs.to("cpu").detach().numpy())
            en_outputs.append(encoder_out_Phvac.to("cpu").detach().numpy())
            de_outputs.append(decoder_out_Phvac.to("cpu").detach().numpy())

        self.to_outputs, self.en_outputs, self.de_outputs = to_outputs, en_outputs, de_outputs
        # De-Norm
        to_out, en_out, de_out = [], [], []
        for idx in range(to_outputs[0].shape[0]):
            to_out.append(loadscal.inverse_transform(to_outputs[0][[idx],:,:].reshape(-1, 1)))
            en_out.append(loadscal.inverse_transform(en_outputs[0][[idx],:,:].reshape(-1, 1)))
            de_out.append(loadscal.inverse_transform(de_outputs[0][[idx],:,:].reshape(-1, 1)))

        self.to_denorm = to_out
        self.en_denorm = en_out
        self.de_denorm = de_out

