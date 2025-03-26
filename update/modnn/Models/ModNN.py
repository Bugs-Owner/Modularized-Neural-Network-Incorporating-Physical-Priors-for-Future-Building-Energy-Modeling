import torch
import torch.nn as nn

# --- Zone Module ---
class zone(nn.Module):
    """
    Assume well-mixed condition, and we have: Q = cmΔTzone
    In other words, ΔTzone = Q/cm, that can be learned by a simple linear layer
    The input of this module is ∑q, each q is calculated by distinct module shown below
    """
    def __init__(self, input_size, output_size):
        super(zone, self).__init__()
        # Set bias to False here is important, since this module is learning a weight only (cm)
        self.FC1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.FC1(x)

# --- Internal Gain Module ---
class internal(nn.Module):
    """
    The internal heat gain module we used here is a simple MLP
    We use it to calculate the q_int from convection ONLY
    There are two ways to consider the internal heat gain from radiation heat transfer
    1) Replace MLP by any type of RNN
    2) Add a lookback window for MLP (for example, use t-N to t steps feature to predict the t step's heat gain)

    The internal heat gain comes from Lighting, Occupant, Appliance, so on so forth
    But they can be represented by a factor (alpha) multiply with a "schedule" (sch), for example:
    q_light = alpha_light * sch_light
    q_human = alpha_human * sch_human
    q_cooking = alpha_cooking * sch_cooking

    If detailed information is available, we can replace it by Physics Equations
    For example, 80~100W * Number_of_people
    Otherwise, we learn form periodic features, such as time of a day, day of a week, in sin/cos form
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(internal, self).__init__()
        self.FC1 = nn.Linear(input_size, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Linear(output_size, output_size, bias=False)

    def forward(self, x):
        x = self.relu(self.FC1(x))
        # x = self.relu(self.FC2(x))
        x = self.sigmoid(self.FC3(x))  # Use sigmoid to bound the output within 0 to 1, kind of norm them as a "schedule"
        return self.scale(x)

# --- HVAC Module ---
class hvac(nn.Module):
    """
    A simplified linear module is used here for AIR SIDE SYSTEM ONLY
    Change it to any type of RNN or add look back window for RADIATION SYSTEM

    The input is pre-calculated pHVAC (thermal load)
    But if the raw data is HVAC energy, or supply air flow/temperature
    No worry, we can add another system module to learn this relation easy
    """
    def __init__(self, input_size, output_size):
        super(hvac, self).__init__()
        self.FC1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.FC1(x)

# --- External Disturbance Module ---
class external(nn.Module):
    """
    We use a LSTM to calculate the external disturbance
    It can switch to any type of RNN or add look back window for MLP to consider solar radiation

    This module is learning the heat transfer through envelop, including conduction, convection and radiation
    And can be seperated into different sub-modules, please select case by case
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(external, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x_input, hidden_state):
        lstm_out, hidden = self.lstm(x_input, hidden_state)
        last_output = lstm_out[:, -1:, :]
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        return out, hidden

# --- Modular Physics-Informed Neural Network ---
class ModNN(nn.Module):
    """
    We form it as a time-stepper model, in other words, we are predict the ΔTzone for each timestep
    """
    def __init__(self, args):
        super(ModNN, self).__init__()
        para = args["para"]
        self.encoLen = args["enLen"]
        self.device = args['device']
        self.window = para["window"]

        self.Ext = external(para["Ext_in"], para["Ext_h"], para["Ext_out"])
        self.Zone = zone(para["Zone_in"], para["Zone_out"])
        self.HVAC = hvac(para["HVAC_in"], para["HVAC_out"])
        self.Int = internal(para["Int_in"], para["Int_h"], para["Int_out"])

    def forward(self, input_X):
        """
        input_X order: [T_zone, T_ambient, solar, day_sin, day_cos, occ, phvac]
        Shape: [batch_size, time_steps, features]
        """
        Ext_X = input_X[:, :, [1, 2, 3, 4]]  # Ambient, Solar, Sin/Cos
        Int_X = input_X[:, :, [3, 4, 5]]     # Sin/Cos + Occupancy
        HVAC_X = input_X[:, :, [6]]          # HVAC power input

        # Initialize outputs
        TOut_list = torch.zeros_like(input_X[:, :, [0]]).to(self.device)
        HVAC_list = torch.zeros_like(HVAC_X).to(self.device)
        Ext_list = torch.zeros_like(HVAC_X).to(self.device)
        Int_list = torch.zeros_like(HVAC_X).to(self.device)

        # Initialize encoder hidden states
        h_ext = torch.ones(1, input_X.shape[0], self.Ext.lstm.hidden_size).to(self.device)
        c_ext = torch.ones(1, input_X.shape[0], self.Ext.lstm.hidden_size).to(self.device)

        # This is the look back window
        # I strongly suggest to use it, if you are using MLP external module
        # The value if how long you want to look back, typically, larger value for heavy structures

        # For RNN type, since the historical info is already stored in the hidden vector,
        # Then it is just a hyperparameter
        window_size = self.window
        for i in range(window_size):
            TOut_list[:, i, :] = input_X[:, i, [0]]
            HVAC_list[:, i, :] = HVAC_X[:, i, :]

        E_Zone_T = input_X[:, [[window_size]], [0]]

        # --- Encoding Phase ---
        # Learn the heat stored in the building due to thermal mass
        # Aim to provide a better initial starting point
        for i in range(window_size, self.encoLen):
            # Self-learning mechanism
            # For each timestep, we need Tzone as input, since it directly impact the heat exchange
            # However, for a forecast problem, Tzone for future is Unknown
            # Which is calculated by previous input, and using a recursive way to get it
            # Therefore we might get accumulative error

            # So we use mixed data (predicted Tzone and measured Tzone) in the training stage
            # To speed up the learning convergency
            # And gradully adapt to predicted only

            ratio = i / self.encoLen
            TOut_list[:, i, :] = E_Zone_T.squeeze(1)
            ext_embed = torch.cat([
                input_X[:, i-window_size+1:i+1, [0]] * ratio + TOut_list[:, i-window_size+1:i+1, :] * (1 - ratio),
                Ext_X[:, i-window_size+1:i+1, :]
            ], dim=2)

            ext_flux, (h_ext, c_ext) = self.Ext(ext_embed, (h_ext, c_ext))
            hvac_flux = HVAC_X[:, i:i+1, :]
            # The internal heat gain module used here Did Not consider Tzone
            # But actually, the heat transfer factor is Tzone dependant
            # For a well controlled space, it's not important
            # But if you are interested in extreme case, please add Tzone for Int input
            int_flux = self.Int(Int_X[:, i:i+1, :])

            total_flux = ext_flux + hvac_flux + int_flux
            HVAC_list[:, i, :] = hvac_flux.squeeze(1)
            Ext_list[:, i, :] = ext_flux.squeeze(1)
            Int_list[:, i, :] = int_flux.squeeze(1)

            # After get total flux, we can predict the ΔTzone, and use residual connection to predict Tzone step by step
            E_Zone_T = ratio * input_X[:, [[i]], [0]] + (1 - ratio) * E_Zone_T + self.Zone(total_flux)

        # --- Decoding Phase ---
        # Just transfer encoder module here
        for i in range(self.encoLen, Ext_X.shape[1]):
            TOut_list[:, i, :] = E_Zone_T.squeeze(1)
            dec_embed = torch.cat([
                TOut_list[:, i-window_size+1:i+1, :],
                Ext_X[:, i-window_size+1:i+1, :]
            ], dim=2)

            ext_flux, (h_ext, c_ext) = self.Ext(dec_embed, (h_ext, c_ext))
            hvac_flux = HVAC_X[:, i:i+1, :]
            int_flux = self.Int(Int_X[:, i:i+1, :])

            total_flux = ext_flux + hvac_flux + int_flux
            HVAC_list[:, i, :] = hvac_flux.squeeze(1)
            Ext_list[:, i, :] = ext_flux.squeeze(1)
            Int_list[:, i, :] = int_flux.squeeze(1)
            # No self-learning during decoder stage
            E_Zone_T = self.Zone(total_flux) + E_Zone_T

        return TOut_list, HVAC_list, (Ext_list, Int_list)
