import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bias=True)
        self.Defc = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.Defc(lstm_out)

        return output, self.hidden

class Baseline(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoLen = args["enLen"]
        self.decoder = LSTM(input_size= 6,
                            hidden_size=24,
                            output_size=1)
        self.device = args['device']

    def forward(self, input_X):
        Decoder_X = input_X[:, self.encoLen:, [1, 2, 3, 4, 5, 6]]
        Current_X_zone = input_X[:, self.encoLen:self.encoLen + 1, [0]]
        outputs, _ = self.decoder(Decoder_X, (Current_X_zone.reshape(1, Decoder_X.shape[0], 1).repeat(1, 1, 24),
                                              Current_X_zone.reshape(1, Decoder_X.shape[0], 1).repeat(1, 1, 24)))

        return torch.cat((input_X[:, :self.encoLen, [0]], outputs), 1), torch.cat((input_X[:, :self.encoLen, [0]], outputs), 1),torch.cat((input_X[:, :self.encoLen, [0]], outputs), 1)
