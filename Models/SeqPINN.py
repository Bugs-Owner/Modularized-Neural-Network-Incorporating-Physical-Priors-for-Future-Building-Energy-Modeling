import torch
import torch.nn as nn


class gru_Linear(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(gru_Linear, self).__init__()
        self.Fc1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.Fc2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedding = self.Fc1(x)
        embedding = self.relu(embedding)
        embedding = self.Fc2(embedding)
        return embedding


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

    def forward(self, x_input, encoder_hidden_states):
        gru_out, self.hidden = self.gru(x_input, encoder_hidden_states)
        output = self.Enfc(gru_out)

        return output, self.hidden



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


class SeqPinn(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.encoLen = para['encoLen']
        self.encoder_external = gru_encoder(input_size=para["encoder_external_in"],
                                            hidden_size=para["encoder_external_h"],
                                            output_size=para["encoder_external_out"])
        self.encoder_internal = gru_encoder(input_size=para["encoder_internal_in"],
                                            hidden_size=para["encoder_internal_h"],
                                            output_size=para["encoder_internal_out"])
        self.encoder_hvac = gru_encoder(input_size=para["encoder_hvac_in"],
                                        hidden_size=para["encoder_hvac_h"],
                                        output_size=para["encoder_hvac_out"])
        self.decoder_external = gru_decoder(input_size=para["decoder_external_in"],
                                            hidden_size=para["decoder_external_h"],
                                            output_size=para["decoder_external_out"])
        self.decoder_internal = gru_decoder(input_size=para["decoder_internal_in"],
                                            hidden_size=para["decoder_internal_h"],
                                            output_size=para["decoder_internal_out"])
        self.decoder_hvac = gru_decoder(input_size=para["decoder_hvac_in"],
                                        hidden_size=para["decoder_hvac_h"],
                                        output_size=para["decoder_hvac_out"])
        self.encoder_out = gru_Linear(input_size=para["En_out_insize"],
                                      hidden_size=para["En_out_hidden"],
                                      output_size=para["En_out_outsize"])
        self.decoder_out = gru_Linear(input_size=para["De_out_insize"],
                                      hidden_size=para["De_out_hidden"],
                                      output_size=para["De_out_outsize"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_X):
        """
        Input_X should have features: 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
        (Setpoint is not required for single zone)

        Note: there are so many ways to construct encoder-current-decoder structure
        seq2seq is most computation efficient, but hard to explain
        one2one is time-consuming but physical meaningful

        in paper, I was using seq2seq encoder and one2one decoder
        the updated version below is one2one encoder and decoder, the purpose here is to keep consistence
        """
        # Encoder
        # External heat gain
        Encoder_X_Ext = input_X[:, :self.encoLen, [1, 2, 3, 4]]
        # Internal heat gain
        Encoder_X_Int = input_X[:, :self.encoLen, [3, 4, 5]]
        # HVAC
        Encoder_X_HVAC = input_X[:, :self.encoLen, [7]]

        # Current
        Current_X_Ext = input_X[:, self.encoLen:self.encoLen + 1, [1, 2, 3, 4]]
        # Internal heat gain
        Current_X_Int = input_X[:, self.encoLen:self.encoLen + 1, [3, 4, 5]]
        # HVAC
        Current_X_HVAC = input_X[:, self.encoLen:self.encoLen + 1, [7]]

        # Decoder
        # External heat gain
        Decoder_X_Ext = input_X[:, self.encoLen + 1:, [1, 2, 3, 4]]
        # Internal heat gain
        Decoder_X_Int = input_X[:, self.encoLen + 1:, [3, 4, 5]]
        # HVAC
        Decoder_X_HVAC = input_X[:, self.encoLen + 1:, [7]]

        #Save
        encoder_output_list = torch.zeros(Encoder_X_Ext.shape[0], Encoder_X_Ext.shape[1], 1).to(self.device)
        decoder_output_list = torch.zeros(Decoder_X_Ext.shape[0], Decoder_X_Ext.shape[1], 1).to(self.device)
        current_output_list = torch.zeros(Current_X_Ext.shape[0], Current_X_Ext.shape[1], 1).to(self.device)

        # Encoder initialize
        encoder_hidden_Ext_ = torch.zeros(1, Encoder_X_Ext.shape[0], self.encoder_external.gru.hidden_size).to(self.device)
        encoder_hidden_hid_Int_ = torch.zeros(1, Encoder_X_Int.shape[0], self.encoder_internal.gru.hidden_size).to(self.device)
        encoder_hidden_hid_HVAC_ = torch.zeros(1, Encoder_X_HVAC.shape[0], self.encoder_hvac.gru.hidden_size).to(self.device)
        Current_X_zone = input_X[:, [[0]], [0]]
        Encoder_X_Ext_ = torch.cat((Current_X_zone, Encoder_X_Ext[:, [0], :]), 2)
        Encoder_X_Int_ = torch.cat((Current_X_zone, Encoder_X_Int[:, [0], :]), 2)
        Encoder_X_HVAC_ = Encoder_X_HVAC[:, [0], :]

        for i in range(Encoder_X_Ext.shape[1] - 1):
            encoder_output_Ext_, encoder_hidden_Ext_ = self.encoder_external(Encoder_X_Ext_, encoder_hidden_Ext_)
            encoder_output_Int_, encoder_hidden_hid_Int_ = self.encoder_internal(Encoder_X_Int_, encoder_hidden_hid_Int_)
            encoder_output_HVAC_, encoder_hidden_hid_HVAC_ = self.encoder_hvac(Encoder_X_HVAC_, encoder_hidden_hid_HVAC_)
            
            encoder_out_embed = torch.cat((encoder_output_Ext_, encoder_output_Int_, encoder_output_HVAC_), 2)
            encoder_output = self.encoder_out(encoder_out_embed)  #Heat Flux deveided by (Cp*M)
            Current_X_zone += encoder_output
            encoder_output_list[:, i, :] = torch.squeeze(Current_X_zone, dim=1)
            i += 1
            Encoder_X_Ext_ = torch.cat((Current_X_zone, Encoder_X_Ext[:, [i], :]), 2)
            Encoder_X_Int_ = torch.cat((Current_X_zone, Encoder_X_Int[:, [i], :]), 2)
            Encoder_X_HVAC_ = Encoder_X_HVAC[:, [i], :]

        encoder_output_Ext_, encoder_hidden_Ext_ = self.encoder_external(Encoder_X_Ext_, encoder_hidden_Ext_)
        encoder_output_Int_, encoder_hidden_hid_Int_ = self.encoder_internal(Encoder_X_Int_, encoder_hidden_hid_Int_)
        encoder_output_HVAC_, encoder_hidden_hid_HVAC_ = self.encoder_hvac(Encoder_X_HVAC_, encoder_hidden_hid_HVAC_)

        encoder_out_embed = torch.cat((encoder_output_Ext_, encoder_output_Int_, encoder_output_HVAC_), 2)
        encoder_output = self.encoder_out(encoder_out_embed)  # Heat Flux deveided by (Cp*M)
        Current_X_zone += encoder_output
        encoder_output_list[:, i, :] = torch.squeeze(Current_X_zone, dim=1)

        #Update current measurment
        Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
        Current_X_Ext_ = torch.cat((Current_X_zone, Current_X_Ext[:, [0], :]), 2)
        Current_X_Int_ = torch.cat((Current_X_zone, Current_X_Int[:, [0], :]), 2)
        Current_X_HVAC_ = Current_X_HVAC[:, [0], :]

        current_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Current_X_Ext_, encoder_hidden_Ext_)
        current_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Current_X_Int_, encoder_hidden_hid_Int_)
        current_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Current_X_HVAC_, encoder_hidden_hid_HVAC_)

        current_out_embed = torch.cat((current_output_Ext_, current_output_Int_, current_output_HVAC_), 2)
        decoder_output = self.decoder_out(current_out_embed)  # Heat Flux deveided by (Cp*M)
        Current_X_zone += decoder_output
        current_output_list[:, :, :] = Current_X_zone

        # Decoder initialize
        Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [0], :]), 2)
        Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [0], :]), 2)
        Decoder_X_HVAC_ = Decoder_X_HVAC[:, [0], :]

        # Decoder Prediction
        for i in range(Decoder_X_Ext.shape[1] - 1):
            decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
            decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_, decoder_hidden_hid_Int_)
            decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_, decoder_hidden_hid_HVAC_)

            decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
            decoder_output = self.decoder_out(decoder_out_embed)  #Heat Flux deveided by (Cp*M)
            Current_X_zone += decoder_output
            decoder_output_list[:, i, :] = torch.squeeze(Current_X_zone, dim=1)
            i += 1
            Decoder_X_Ext_ = torch.cat((Current_X_zone, Decoder_X_Ext[:, [i], :]), 2)
            Decoder_X_Int_ = torch.cat((Current_X_zone, Decoder_X_Int[:, [i], :]), 2)
            Decoder_X_HVAC_ = Decoder_X_HVAC[:, [i], :]

        decoder_output_Ext_, decoder_hidden_Ext_ = self.decoder_external(Decoder_X_Ext_, decoder_hidden_Ext_)
        decoder_output_Int_, decoder_hidden_hid_Int_ = self.decoder_internal(Decoder_X_Int_, decoder_hidden_hid_Int_)
        decoder_output_HVAC_, decoder_hidden_hid_HVAC_ = self.decoder_hvac(Decoder_X_HVAC_, decoder_hidden_hid_HVAC_)

        decoder_out_embed = torch.cat((decoder_output_Ext_, decoder_output_Int_, decoder_output_HVAC_), 2)
        decoder_output = self.decoder_out(decoder_out_embed)  # Heat Flux deveided by (Cp*M)
        Current_X_zone += decoder_output
        decoder_output_list[:, i, :] = torch.squeeze(Current_X_zone, dim=1)

        outputs = torch.cat((encoder_output_list, current_output_list, decoder_output_list), 1)
        return outputs


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bias=False)
        self.Defc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)

    def forward(self, x_input, encoder_hidden_states):
        gru_out, self.hidden = self.gru(x_input, encoder_hidden_states)
        output = self.Defc(gru_out)

        return output, self.hidden


class Baseline(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.encoLen = para['encoLen']
        self.decoder = LSTM(input_size=6,
                            hidden_size=24,
                            output_size=1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_X):
        """
        Baseline: LSTM
        """
        Decoder_X = input_X[:, self.encoLen:, [1, 2, 3, 4, 5, 7]]
        Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
        outputs, _ = self.decoder(Decoder_X, (Current_X_zone.reshape(1, Decoder_X.shape[0], 1).repeat(1, 1, 24),
                                              Current_X_zone.reshape(1, Decoder_X.shape[0], 1).repeat(1, 1, 24)))

        return outputs

