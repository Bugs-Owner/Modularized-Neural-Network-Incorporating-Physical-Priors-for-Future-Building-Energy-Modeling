import torch
import torch.nn as nn
import numpy as np

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
        self.scale = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.scale(x)

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
        self.FC1 = nn.Linear(input_size-1, hidden_size)
        self.FC2 = nn.Linear(hidden_size, output_size)
        self.scale = nn.Linear(1, 1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding = self.FC1(x[:, :, :-1])
        embedding = self.relu(embedding)
        embedding = self.FC2(embedding)
        embedding = self.sigmoid(embedding) #Force output to 0--1
        embedding = embedding + x[:, :, -1:]
        return self.scale(embedding)

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
        self.scale = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.scale(x)

# --- External Disturbance Module ---
class external(nn.Module):
    """
    We use RNN to calculate the external disturbance.
    RNN allows easy positive constraints compared to GRU/LSTM.
    For disturbance prediction, similar distributions exist during training,
    so strict constraints may not be necessary.

    This module learns heat transfer through the envelope, including conduction, convection, and radiation,
    and can be separated into different sub-modules (choose case by case).
    """
    def __init__(self, input_size, hidden_size, output_size, model_type, window_size, num_layers=1):
        super(external, self).__init__()
        self.timeFC1 = nn.Linear(in_features=2, out_features=2, bias=True)
        self.timeFC2 = nn.Linear(in_features=2, out_features=1, bias=True)

        self.trans_coeff = nn.Parameter(torch.rand(1))
        self.absor_coeff = nn.Parameter(torch.rand(1))

        self.FC = nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        self.tran_FC = nn.Linear(in_features=window_size, out_features=1, bias=True)
        self.conduction = nn.Linear(1, 1, bias=False)
        self.relu = nn.ReLU()

        if model_type == "RNN":
            self.ext_mdl = nn.RNN(input_size=output_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True, bias=True)
        elif model_type == "LSTM":
            self.ext_mdl = nn.LSTM(input_size=output_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bias=True)
        else:
            raise ValueError("Module error, please choose 'RNN' or 'LSTM'.")

    def forward(self, x_input, hidden_state):
        # Structure of x_input
        # Room[0], Ambient[1], Solar[2], Sin/Cos[3,4]
        # Solar radiation adjustment
        # For each surface, the radiation is time related due to solar angle
        # So we learn a time feature first
        time_depend_scale = self.timeFC1(x_input[:, :, [3, 4]])
        time_depend_scale = self.relu(time_depend_scale)
        time_depend_scale = self.relu(self.timeFC2(time_depend_scale))
        adj_sol = x_input[:, :, [2]]

        # Transmittance and Absorptance coefficients
        # Some solar absorbed by ext surfaces, some transmit into space directly
        trans_coeff = torch.sigmoid(self.trans_coeff)
        absor_coeff = torch.sigmoid(self.absor_coeff)

        # Solar transmission and absorption
        trans_sol = trans_coeff * adj_sol
        absor_sol = absor_coeff * adj_sol

        # "Conduction, T_amb-T_room", (actually this process include convection, conduction and radiation)
        temp_diff = x_input[:, :, [1]] - x_input[:, :, [0]]
        q_con = self.conduction(temp_diff)

        # External model
        # Heat through opeque envelop
        ext_input = absor_sol + q_con
        out, hidden = self.ext_mdl(ext_input, hidden_state)
        last_output = out[:, [-1], :]
        # Final gain need to plus the heat through transparent envelop
        output = self.FC(last_output) + trans_sol[:, [-1], :]

        return output, hidden

# --- Modular Physics-Informed Neural Network ---
class ModNN(nn.Module):
    """
    We form it as a time-stepper model, in other words, we are predict the ΔTzone for each timestep
    """
    def __init__(self, args):
        super(ModNN, self).__init__()
        para = args["para"]
        self.encoLen = args["enLen"]
        self.device  = args['device']
        self.window  = para["window"]
        self.ext_mdl = args["ext_mdl"]

        self.Ext = external(para["Ext_in"], para["Ext_h"], para["Ext_out"], self.ext_mdl, para["window"])
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
        deltaQ_list = torch.zeros_like(HVAC_X).to(self.device)
        # Initialize encoder hidden states
        if self.ext_mdl == "RNN":
            ext_hidd = torch.ones(1, input_X.shape[0], self.Ext.ext_mdl.hidden_size).to(self.device)
        else:
            ext_hidd = torch.ones(1, Ext_X.shape[0], self.Ext.ext_mdl.hidden_size).to(self.device)
            ext_cell = torch.ones(1, Ext_X.shape[0], self.Ext.ext_mdl.hidden_size).to(self.device)
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

            # To speed up the learning convergency,
            # We use mixed data (predicted Tzone and measured Tzone) in the training stage
            ratio = i/self.encoLen
            TOut_list[:, i, :] = E_Zone_T.squeeze(1)
            ext_embed = torch.cat([
                input_X[:, i-window_size+1:i+1, [0]] * ratio + TOut_list[:, i-window_size+1:i+1, :] * (1 - ratio),
                Ext_X[:, i-window_size+1:i+1, :]
            ], dim=2)
            if self.ext_mdl == "RNN":
                ext_flux, ext_hidd = self.Ext(ext_embed, ext_hidd)
            else:
                ext_flux, (ext_hidd, ext_cell) = self.Ext(ext_embed, (ext_hidd, ext_cell))
            hvac_flux = self.HVAC(HVAC_X[:, i:i+1, :])
            # The internal heat gain module used here Did Not consider Tzone
            # But actually, the heat transfer factor is Tzone dependant
            # For a well controlled space, it's not important
            # But if you are interested in extreme case, please add Tzone for Int input
            int_flux = self.Int(Int_X[:, i:i+1, :])
            total_flux = ext_flux + hvac_flux + int_flux
            HVAC_list[:, i, :] = hvac_flux.squeeze(1)
            Ext_list[:, i, :] = ext_flux.squeeze(1)
            Int_list[:, i, :] = int_flux.squeeze(1)
            deltaQ_list[:, i, :] = total_flux.squeeze(1)

            # After get total flux, we can predict the ΔTzone, and use residual connection to predict Tzone step by step
            E_Zone_T =  E_Zone_T + self.Zone(total_flux)

        # --- Decoding Phase ---
        # Just transfer encoder module here
        for i in range(self.encoLen, Ext_X.shape[1]):
            TOut_list[:, i, :] = E_Zone_T.squeeze(1)
            ext_embed = torch.cat([
                TOut_list[:, i-window_size+1:i+1, :],
                Ext_X[:, i-window_size+1:i+1, :]
            ], dim=2)

            if self.ext_mdl == "RNN":
                ext_flux, ext_hidd = self.Ext(ext_embed, ext_hidd)
            else:
                ext_flux, (ext_hidd, ext_cell) = self.Ext(ext_embed, (ext_hidd, ext_cell))
            hvac_flux = self.HVAC(HVAC_X[:, i:i+1, :])
            int_flux = self.Int(Int_X[:, i:i+1, :])
            total_flux = ext_flux + hvac_flux + int_flux

            HVAC_list[:, i, :] = hvac_flux.squeeze(1)
            Ext_list[:, i, :] = ext_flux.squeeze(1)
            Int_list[:, i, :] = int_flux.squeeze(1)
            deltaQ_list[:, i, :] = total_flux.squeeze(1)
            # No self-learning during decoder stage
            E_Zone_T = E_Zone_T + self.Zone(total_flux)

        # Return not only temperature, but also latent flux
        # Not sure how to use these fluxes right now, but leave for placeholder
        return TOut_list, HVAC_list, (Ext_list, Int_list, deltaQ_list)