import torch
import torch.nn as nn

class LSTM_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bias=True)
        self.Enfc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        output = self.relu(self.Enfc(lstm_out))*(-1)

        return output, self.hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class LSTM_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bias=True)
        self.Defc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.relu(self.Defc(lstm_out))*-1

        return output, self.hidden


class SolNN(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.encoder = LSTM_encoder(input_size=para["SolNN_encoder_insize"],
                                    hidden_size=para["SolNN_encoder_hidden"],
                                    output_size=para["SolNN_decoder_outsize"])
        self.decoder = LSTM_decoder(input_size=para["SolNN_decoder_insize"],
                                    hidden_size=para["SolNN_decoder_hidden"],
                                    output_size=para["SolNN_decoder_outsize"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoLen = para['encoLen']

    def forward(self, input_X):
        # 0.Tzone; 1:Tamb; 2:Solar; 3:Day_sin; 4:Day_cos; 5.Occ; 6.Tset; 7.Phvac;
        # 8.upperscaled; 9.lowerscaled; 10.weight; 11.price
        # Encoder
        Encoder_X = input_X[:, :self.encoLen, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]]
        # Decoder
        Decoder_X = input_X[:, self.encoLen:, [1, 2, 3, 4, 5, 8, 9, 10, 11]]
        encoder_out, encoder_hidden = self.encoder(Encoder_X)
        decoder_out, _ = self.decoder(Decoder_X, encoder_hidden_states = encoder_hidden)
        outputs = torch.cat((encoder_out, decoder_out), 1)
        return outputs
