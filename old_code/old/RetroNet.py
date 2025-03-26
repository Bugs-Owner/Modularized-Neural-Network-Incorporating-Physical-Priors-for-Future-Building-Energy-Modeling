import torch
import torch.nn as nn
import numpy as np
from tqdm import trange


class gru_Linear(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(gru_Linear, self).__init__()
        self.Fc1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.Fc2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        Embedding_state = self.Fc1(x)
        Embedding_state = self.relu(Embedding_state)
        Embedding_state = self.Fc2(Embedding_state)
        return Embedding_state

class Reluliner(nn.Module):
    def __init__(self):
        super(Reluliner, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x)*(-1)

# class gru_Linear(nn.Module):
#
#     def __init__(self, input_size, hidden_size, output_size):
#         super(gru_Linear, self).__init__()
#         self.Fc1 = nn.Linear(in_features=1, out_features=8, bias=True)
#         self.Fc2 = nn.Linear(in_features=8, out_features=1, bias=True)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         Embedding_state = self.Fc1(x)
#         Embedding_state = self.relu(Embedding_state)
#         Embedding_state = self.Fc2(Embedding_state)
#         return Embedding_state

class gru_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(gru_encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # define GRU layer
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bias=False)
        self.Enfc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)

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
                          num_layers=num_layers, batch_first=True, bias=False)
        self.Defc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)

    def forward(self, x_input, encoder_hidden_states):
        gru_out, self.hidden = self.gru(x_input, encoder_hidden_states)
        output = self.Defc(gru_out)

        return output, self.hidden


class gru_seq2seq(nn.Module):
    def __init__(self, para):
        super(gru_seq2seq, self).__init__()
        self.positive=Reluliner()
        self.Surf_encoder = gru_encoder(input_size=para["Surf_in_enco"], hidden_size=para["Surf_h_enco"],
                                        output_size=para["Surf_out_enco"])
        self.Surf_decoder = gru_decoder(input_size=para["Surf_in_deco"], hidden_size=para["Surf_h_deco"],
                                        output_size=para["Surf_out_deco"])
        self.Cond_encoder = gru_Linear(input_size=para["Cond_in_enco"], hidden_size=para["Cond_h_enco"],
                                        output_size=para["Cond_out_enco"])
        self.Cond_decoder = gru_Linear(input_size=para["Cond_in_deco"], hidden_size=para["Cond_h_deco"],
                                        output_size=para["Cond_out_deco"])
        self.SRad_encoder = gru_encoder(input_size=para["SRad_in_enco"], hidden_size=para["SRad_h_enco"],
                                        output_size=para["SRad_out_enco"])
        self.SRad_decoder = gru_decoder(input_size=para["SRad_in_deco"], hidden_size=para["SRad_h_deco"],
                                        output_size=para["SRad_out_deco"])
        self.IRad_encoder = gru_encoder(input_size=para["IRad_in_enco"], hidden_size=para["IRad_h_enco"],
                                        output_size=para["IRad_out_enco"])
        self.IRad_decoder = gru_decoder(input_size=para["IRad_in_deco"], hidden_size=para["IRad_h_deco"],
                                        output_size=para["IRad_out_deco"])
        self.Inf_encoder = gru_encoder(input_size=para["Inf_in_enco"], hidden_size=para["Inf_h_enco"],
                                       output_size=para["Inf_out_enco"])
        self.Inf_decoder = gru_decoder(input_size=para["Inf_in_deco"], hidden_size=para["Inf_h_deco"],
                                       output_size=para["Inf_out_deco"])
        self.Conv_encoder = gru_Linear(input_size=para["Conv_in_enco"], hidden_size=para["Conv_h_enco"],
                                       output_size=para["Conv_out_enco"])
        self.Conv_decoder = gru_Linear(input_size=para["Conv_in_deco"], hidden_size=para["Conv_h_deco"],
                                       output_size=para["Conv_out_deco"])
        self.Phvac_encoder = gru_Linear(input_size=para["Phvac_in_enco"], hidden_size=para["Phvac_h_enco"],
                                       output_size=para["Phvac_out_enco"])
        self.Phvac_decoder = gru_Linear(input_size=para["Phvac_in_deco"], hidden_size=para["Phvac_h_deco"],
                                       output_size=para["Phvac_out_deco"])

        self.lr = para["lr"]
        self.epoch = para["epochs"]
        self.encoLen = para["encoLen"]
        self.decoLen = para["decoLen"]

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
                    optimizer.zero_grad()

                    # 1. Conduction
                    Enco_Surf_X = input_X[:, :self.encoLen, [1, 2, 3, 4]]
                    Deco_Surf_X = input_X[:, self.encoLen:, [1, 2, 3, 4]]
                    # 2. Solar Radiation
                    Enco_SRad_X = input_X[:, :self.encoLen, [2, 3, 4, 6]]
                    Deco_SRad_X = input_X[:, self.encoLen:, [2, 3, 4, 6]]
                    # 3. Internal Radiation
                    Enco_IRad_X = input_X[:, :self.encoLen, [3, 4, 5, 6]]
                    Deco_IRad_X = input_X[:, self.encoLen:, [3, 4, 5, 6]]
                    # 4. Infiltration
                    Enco_Infil_X = input_X[:, :self.encoLen, [1, 6]]
                    Deco_Infil_X = input_X[:, self.encoLen:, [1, 6]]

                    # 1. Conduction Calculation
                    enco_out_Surf, enco_hid_Surf = self.Surf_encoder(Enco_Surf_X)
                    deco_out_Surf, deco_hid_Surf = self.Surf_decoder(Deco_Surf_X, enco_hid_Surf)
                    enco_out_Cond = self.Cond_encoder(torch.cat((enco_out_Surf, input_X[:, :self.encoLen, [6]]), 2))
                    deco_out_Cond = self.Cond_decoder(torch.cat((deco_out_Surf, input_X[:, self.encoLen:, [6]]), 2))

                    # 2. Radiation Calculation
                    enco_out_SRad, enco_hid_SRad = self.SRad_encoder(Enco_SRad_X)
                    deco_out_SRad, deco_hid_SRad = self.SRad_decoder(Deco_SRad_X, enco_hid_SRad)
                    enco_out_SRad = self.positive(enco_out_SRad)
                    deco_out_SRad = self.positive(deco_out_SRad)
                    # 3. Encoder Inter Radiation
                    enco_out_IRad, enco_hid_IRad = self.IRad_encoder(Enco_IRad_X)
                    deco_out_IRad, deco_hid_IRad = self.IRad_decoder(Deco_IRad_X, enco_hid_IRad)
                    # 4. Infiltration
                    enco_out_Inf, enco_hid_Inf = self.Inf_encoder(Enco_Infil_X)
                    deco_out_Inf, deco_hid_Inf = self.Inf_decoder(Deco_Infil_X, enco_hid_Inf)

                    enco_conv_embed = torch.cat((enco_out_Cond, enco_out_SRad, enco_out_IRad), 2)
                    deco_conv_embed = torch.cat((deco_out_Cond, deco_out_SRad, deco_out_IRad), 2)
                    # enco_conv_embed = enco_out_Cond + enco_out_SRad + enco_out_IRad
                    # deco_conv_embed = deco_out_Cond + deco_out_SRad + deco_out_IRad
                    # enco_out_Conv,_ = self.Conv_encoder(enco_conv_embed)
                    # deco_out_Conv,nan = self.Conv_decoder(deco_conv_embed,_)
                    enco_out_Conv = self.Conv_encoder(enco_conv_embed)
                    deco_out_Conv = self.Conv_decoder(deco_conv_embed)
                    enco_hvac_embed = torch.cat((enco_out_Conv, enco_out_Inf, enco_out_IRad), 2)
                    deco_hvac_embed = torch.cat((deco_out_Conv, deco_out_Inf, deco_out_IRad), 2)
                    # enco_hvac_embed = enco_out_Conv + enco_out_Inf + enco_out_IRad
                    # deco_hvac_embed = deco_out_Conv + deco_out_Inf + deco_out_IRad
                    # enco_out_Phvac,hid = self.Phvac_encoder(enco_hvac_embed)
                    # deco_out_Phvac,nan = self.Phvac_decoder(deco_hvac_embed,hid)
                    enco_out_Phvac = self.Phvac_encoder(enco_hvac_embed)
                    deco_out_Phvac = self.Phvac_decoder(deco_hvac_embed)
                    enco_out_Phvac = self.positive(enco_out_Phvac)
                    deco_out_Phvac = self.positive(deco_out_Phvac)

                    decoder_loss, encoder_loss, total_loss = 0., 0., 0.
                    outputs = torch.cat((enco_out_Phvac, deco_out_Phvac), 1)
                    deloss = criterion(deco_out_Phvac, Decoder_y)
                    enloss = criterion(enco_out_Phvac, Encoder_y)
                    toloss = criterion(outputs, output_y)

                    decoder_loss += deloss.item()
                    encoder_loss += enloss.item()
                    total_loss += toloss.item()

                    # backpropagation
                    toloss.backward()
                    optimizer.step()

                    # for p in self.Phvac_decoder.Fc1.parameters():
                    #     p.data[:, [0]].clamp_(0)  # First row
                    # for p in self.Phvac_encoder.Fc1.parameters():
                    #     p.data[:, [0]].clamp_(0)  # First row
                    # for p in self.Conv_encoder.Fc1.parameters():
                    #     p.data=torch.clamp(p.data, max=0)
                    # for p in self.Conv_decoder.Fc1.parameters():
                    #     p.data=torch.clamp(p.data, max=0)

                    for p in self.Conv_encoder.Fc1.parameters():
                        p.data=torch.clamp(p.data, min=0)
                    for p in self.Conv_decoder.Fc1.parameters():
                        p.data=torch.clamp(p.data, min=0)
                    for p in self.Conv_encoder.Fc2.parameters():
                        p.data=torch.clamp(p.data, min=0)
                    for p in self.Conv_decoder.Fc2.parameters():
                        p.data=torch.clamp(p.data, min=0)
                    for p in self.Phvac_encoder.parameters():
                        p.data=torch.clamp(p.data, min=0)
                    for p in self.Phvac_decoder.parameters():
                        p.data=torch.clamp(p.data, min=0)

                    # for p in self.Conv_encoder.Fc1.parameters():
                    #     torch.clamp(p.data[:, [1]], max=0)
                    # for p in self.Conv_decoder.Fc1.parameters():
                    #     torch.clamp(p.data[:, [1]], max=0)
                    # for p in self.Conv_encoder.Fc2.parameters():
                    #     p.data.clamp_(0)  # First row
                    # for p in self.Conv_decoder.Fc2.parameters():
                    #     p.data.clamp_(0)  # First row
                    # for p in self.Phvac_encoder.Fc1.parameters():
                    #     p.data[:, [0]].clamp_(0)  # F
                    # for p in self.Phvac_decoder.Fc1.parameters():
                    #     p.data[:, [0]].clamp_(0)
                    # for p in self.Phvac_encoder.Fc2.parameters():
                    #     p.data.clamp_(0)  # F
                    # for p in self.Phvac_decoder.Fc2.parameters():
                    #     p.data.clamp_(0)
                    # for p in self.Conv_encoder.Fc1.parameters():
                    #     torch.clamp(p.data[:, [1]], max=0)
                    # for p in self.Conv_decoder.Fc1.parameters():
                    #     torch.clamp(p.data[:, [1]], max=0)
                    # for p in self.Phvac_decoder.Fc2.parameters():
                    #     torch.clamp(p.data.clamp_(0), max=0)
                    # for p in self.Phvac_encoder.Fc2.parameters():
                    #     torch.clamp(p.data.clamp_(0), max=0)
                    # for p in self.Conv_encoder.Fc2.parameters():
                    #     torch.clamp(p.data.clamp_(0), max=0)
                    # for p in self.Conv_decoder.Fc2.parameters():
                    #     torch.clamp(p.data.clamp_(0), max=0)

                encoder_loss /= n_batches
                decoder_loss /= n_batches
                total_loss /= n_batches

                enlosses[it] = encoder_loss
                delosses[it] = decoder_loss
                tolosses[it] = total_loss

                tr.set_postfix(encoder_loss="{0:.6f}".format(encoder_loss), decoder_loss="{0:.6f}".format(decoder_loss),
                               total_loss="{0:.6f}".format(total_loss))

    def test_model(self, dataloder, hvacscal):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            # Divide input_X to Encoder_X and Decoder_X
            # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
            # 1. Conduction
            Enco_Surf_X = input_X[:, :self.encoLen, [1, 2, 3, 4]]
            Deco_Surf_X = input_X[:, self.encoLen:, [1, 2, 3, 4]]
            # 2. Solar Radiation
            Enco_SRad_X = input_X[:, :self.encoLen, [2, 3, 4, 6]]
            Deco_SRad_X = input_X[:, self.encoLen:, [2, 3, 4, 6]]
            # 3. Internal Radiation
            Enco_IRad_X = input_X[:, :self.encoLen, [3, 4, 5, 6]]
            Deco_IRad_X = input_X[:, self.encoLen:, [3, 4, 5, 6]]
            # 4. Infiltration
            Enco_Infil_X = input_X[:, :self.encoLen, [1, 6]]
            Deco_Infil_X = input_X[:, self.encoLen:, [1, 6]]

            # 1. Conduction Calculation
            enco_out_Surf, enco_hid_Surf = self.Surf_encoder(Enco_Surf_X)
            deco_out_Surf, deco_hid_Surf = self.Surf_decoder(Deco_Surf_X, enco_hid_Surf)
            enco_out_Cond = self.Cond_encoder(torch.cat((enco_out_Surf, input_X[:, :self.encoLen, [6]]), 2))
            deco_out_Cond = self.Cond_decoder(torch.cat((deco_out_Surf, input_X[:, self.encoLen:, [6]]), 2))

            # 2. Radiation Calculation
            enco_out_SRad, enco_hid_SRad = self.SRad_encoder(Enco_SRad_X)
            deco_out_SRad, deco_hid_SRad = self.SRad_decoder(Deco_SRad_X, enco_hid_SRad)
            enco_out_SRad = self.positive(enco_out_SRad)
            deco_out_SRad = self.positive(deco_out_SRad)

            # 3. Encoder Inter Radiation
            enco_out_IRad, enco_hid_IRad = self.IRad_encoder(Enco_IRad_X)
            deco_out_IRad, deco_hid_IRad = self.IRad_decoder(Deco_IRad_X, enco_hid_IRad)
            # 4. Infiltration
            enco_out_Inf, enco_hid_Inf = self.Inf_encoder(Enco_Infil_X)
            deco_out_Inf, deco_hid_Inf = self.Inf_decoder(Deco_Infil_X, enco_hid_Inf)

            enco_conv_embed = torch.cat((enco_out_Cond, enco_out_SRad, enco_out_IRad), 2)
            deco_conv_embed = torch.cat((deco_out_Cond, deco_out_SRad, deco_out_IRad), 2)
            # enco_conv_embed = enco_out_Cond + enco_out_SRad + enco_out_IRad
            # deco_conv_embed = deco_out_Cond + deco_out_SRad + deco_out_IRad
            # enco_out_Conv,_ = self.Conv_encoder(enco_conv_embed)
            # deco_out_Conv,nan = self.Conv_decoder(deco_conv_embed,_)
            enco_out_Conv = self.Conv_encoder(enco_conv_embed)
            deco_out_Conv = self.Conv_decoder(deco_conv_embed)
            enco_hvac_embed = torch.cat((enco_out_Conv, enco_out_Inf, enco_out_IRad), 2)
            deco_hvac_embed = torch.cat((deco_out_Conv, deco_out_Inf, deco_out_IRad), 2)
            # enco_hvac_embed = enco_out_Conv + enco_out_Inf + enco_out_IRad
            # deco_hvac_embed = deco_out_Conv + deco_out_Inf + deco_out_IRad
            # enco_out_Phvac,hid = self.Phvac_encoder(enco_hvac_embed)
            # deco_out_Phvac,nan = self.Phvac_decoder(deco_hvac_embed,hid)
            enco_out_Phvac = self.Phvac_encoder(enco_hvac_embed)
            deco_out_Phvac = self.Phvac_decoder(deco_hvac_embed)
            enco_out_Phvac = self.positive(enco_out_Phvac)
            deco_out_Phvac = self.positive(deco_out_Phvac)

            # De-Norm
            de_out = []
            conduction, convection = [], []
            infiltration, Iradiation, Sradiation= [], [], []
            for idx in range(deco_out_Phvac.shape[0]):
                de_out.append(hvacscal.inverse_transform(deco_out_Phvac[idx].reshape(-1, 1).cpu().detach().numpy()))
                conduction.append(hvacscal.inverse_transform(deco_out_Cond[idx].reshape(-1, 1).cpu().detach().numpy()))
                convection.append(hvacscal.inverse_transform(deco_out_Conv[idx].reshape(-1, 1).cpu().detach().numpy()))
                infiltration.append(hvacscal.inverse_transform(deco_out_Inf[idx].reshape(-1, 1).cpu().detach().numpy()))
                Iradiation.append(hvacscal.inverse_transform(deco_out_IRad[idx].reshape(-1, 1).cpu().detach().numpy()))
                Sradiation.append(hvacscal.inverse_transform(deco_out_SRad[idx].reshape(-1, 1).cpu().detach().numpy()))

            self.de_denorm = de_out
            self.conduction = conduction
            self.convection = convection
            self.infiltration = infiltration
            self.Iradiation = Iradiation
            self.Sradiation = Sradiation

    def retro_model(self, dataloder, hvacscal, retrolist):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for input_X, output_y in dataloder:
            input_X, output_y = input_X.to(device), output_y.to(device)
            # Divide input_X to Encoder_X and Decoder_X
            # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;

            # 1. Conduction
            Enco_Surf_X = input_X[:, :self.encoLen, [1, 2, 3, 4]]
            Deco_Surf_X = input_X[:, self.encoLen:, [1, 2, 3, 4]]
            # 2. Solar Radiation
            Enco_SRad_X = input_X[:, :self.encoLen, [2, 3, 4, 6]]
            Deco_SRad_X = input_X[:, self.encoLen:, [2, 3, 4, 6]]
            # 3. Internal Radiation
            Enco_IRad_X = input_X[:, :self.encoLen, [3, 4, 5, 6]]
            Deco_IRad_X = input_X[:, self.encoLen:, [3, 4, 5, 6]]
            # 4. Infiltration
            Enco_Infil_X = input_X[:, :self.encoLen, [1, 6]]
            Deco_Infil_X = input_X[:, self.encoLen:, [1, 6]]

            # 1. Conduction Calculation
            enco_out_Surf, enco_hid_Surf = self.Surf_encoder(Enco_Surf_X)
            deco_out_Surf, deco_hid_Surf = self.Surf_decoder(Deco_Surf_X, enco_hid_Surf)
            enco_out_Cond = self.Cond_encoder(torch.cat((enco_out_Surf, input_X[:, :self.encoLen, [6]]), 2))
            deco_out_Cond = self.Cond_decoder(torch.cat((deco_out_Surf, input_X[:, self.encoLen:, [6]]), 2))

            # 2. Radiation Calculation
            enco_out_SRad, enco_hid_SRad = self.SRad_encoder(Enco_SRad_X)
            deco_out_SRad, deco_hid_SRad = self.SRad_decoder(Deco_SRad_X, enco_hid_SRad)
            enco_out_SRad = self.positive(enco_out_SRad)
            deco_out_SRad = self.positive(deco_out_SRad)
            # 3. Encoder Inter Radiation
            enco_out_IRad, enco_hid_IRad = self.IRad_encoder(Enco_IRad_X)
            deco_out_IRad, deco_hid_IRad = self.IRad_decoder(Deco_IRad_X, enco_hid_IRad)
            # 4. Infiltration
            enco_out_Inf, enco_hid_Inf = self.Inf_encoder(Enco_Infil_X)
            deco_out_Inf, deco_hid_Inf = self.Inf_decoder(Deco_Infil_X, enco_hid_Inf)

            enco_conv_embed = torch.cat((enco_out_Cond*retrolist['wall'], enco_out_SRad*retrolist['SHGC'], enco_out_IRad), 2)
            deco_conv_embed = torch.cat((deco_out_Cond*retrolist['wall'], deco_out_SRad*retrolist['SHGC'], deco_out_IRad), 2)
            # enco_conv_embed = enco_out_Cond + enco_out_SRad + enco_out_IRad
            # deco_conv_embed = deco_out_Cond + deco_out_SRad + deco_out_IRad
            # enco_out_Conv,_ = self.Conv_encoder(enco_conv_embed)
            # deco_out_Conv,nan = self.Conv_decoder(deco_conv_embed,_)
            enco_out_Conv = self.Conv_encoder(enco_conv_embed)
            deco_out_Conv = self.Conv_decoder(deco_conv_embed)
            enco_hvac_embed = torch.cat((enco_out_Conv, enco_out_Inf, enco_out_IRad), 2)
            deco_hvac_embed = torch.cat((deco_out_Conv, deco_out_Inf, deco_out_IRad), 2)
            # enco_hvac_embed = enco_out_Conv + enco_out_Inf + enco_out_IRad
            # deco_hvac_embed = deco_out_Conv + deco_out_Inf + deco_out_IRad
            # enco_out_Phvac,hid = self.Phvac_encoder(enco_hvac_embed)
            # deco_out_Phvac,nan = self.Phvac_decoder(deco_hvac_embed,hid)
            enco_out_Phvac = self.Phvac_encoder(enco_hvac_embed)
            deco_out_Phvac = self.Phvac_decoder(deco_hvac_embed)
            enco_out_Phvac = self.positive(enco_out_Phvac)
            deco_out_Phvac = self.positive(deco_out_Phvac)

            # De-Norm
            de_out = []
            conduction, convection = [], []
            infiltration, Iradiation, Sradiation= [], [], []
            for idx in range(deco_out_Phvac.shape[0]):
                de_out.append(hvacscal.inverse_transform(deco_out_Phvac[idx].reshape(-1, 1).cpu().detach().numpy()))
                conduction.append(hvacscal.inverse_transform(deco_out_Cond[idx].reshape(-1, 1).cpu().detach().numpy()))
                convection.append(hvacscal.inverse_transform(deco_out_Conv[idx].reshape(-1, 1).cpu().detach().numpy()))
                infiltration.append(hvacscal.inverse_transform(deco_out_Inf[idx].reshape(-1, 1).cpu().detach().numpy()))
                Iradiation.append(hvacscal.inverse_transform(deco_out_IRad[idx].reshape(-1, 1).cpu().detach().numpy()))
                Sradiation.append(hvacscal.inverse_transform(deco_out_SRad[idx].reshape(-1, 1).cpu().detach().numpy()))

            self.Retrode_denorm = de_out
            self.Retroconduction = conduction
            self.Retroconvection = convection
            self.Retroinfiltration = infiltration
            self.RetroIradiation = Iradiation
            self.RetroSradiation = Sradiation




