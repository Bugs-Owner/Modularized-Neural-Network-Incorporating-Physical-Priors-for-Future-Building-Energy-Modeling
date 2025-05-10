import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Solar Radiation Module ---
class SolarRadiation(nn.Module):
    def __init__(self, envelop, name="Solar"):
        super().__init__()
        # Initialization
        self.envelop = envelop

        # Load transmission and absorption coefficient
        self.direct_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.abs_wall = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.abs_roof = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.name = name

    def forward(self, radiation, n_wall, n_roof):
        """
        Calculate solar heat gains [transmission and absorption] in a vectorized way.

        Args:
            radiation: Solar radiation input [batch, time, 1] or [batch, time, directions]
                      (e.g., [batch, time, 4] for E,S,W,N directions)
            n_wall: Number of wall components
            n_roof: Number of roof components

        Returns:
            direct_gain: Heat directly entering zone
            wall_gains: List of absorbed heat for each wall [n_wall elements]
            roof_gains: List of absorbed heat for each roof [n_roof elements]
        """
        # Apply softplus to ensure positive coefficients
        direct_coef = F.softplus(self.direct_gain) * self.envelop["direct_coef"]
        abs_wall_coef = F.softplus(self.abs_wall) * self.envelop["abs_wall_coef"]
        abs_roof_coef = F.softplus(self.abs_roof) * self.envelop["abs_roof_coef"]

        # Transmission solar gain
        direct_gain = direct_coef * radiation

        # Absorption solar gain
        # Wall
        batch_size, time_steps = radiation.shape[0], radiation.shape[1]
        if radiation.shape[-1] > 1:  # TODO: add solar angle model to distribute solar radiations
            # Handle multi-directional radiation
            wall_gains = []
            for i in range(n_wall):
                dir_idx = min(i, radiation.shape[-1] - 1)
                wall_gain = abs_wall_coef * radiation[:, :, dir_idx:dir_idx + 1]
                wall_gains.append(wall_gain)
        else:
            wall_gain = abs_wall_coef * radiation
            wall_gains = [wall_gain] * n_wall

        # Roof
        roof_gain = abs_roof_coef * radiation
        roof_gains = [roof_gain] * n_roof

        return direct_gain, wall_gains, roof_gains


# --- Envelope Module [RC, R] ---
class EnvelopeRC(nn.Module):
    def __init__(self, envelop, n_wall, n_roof, n_window):
        super().__init__()
        self.n_wall = n_wall
        self.n_roof = n_roof
        self.n_window = n_window
        self.n_rc = n_wall + n_roof
        self.envelop = envelop

        # Initialization
        wall_r_init = torch.tensor([1.0 / (10.0 + i * 1.0) for i in range(n_wall)], dtype=torch.float32)
        roof_r_init = torch.tensor([1.0 / (20.0 + i * 2.0) for i in range(n_roof)], dtype=torch.float32)
        self.rc_R_inv = nn.Parameter(torch.cat([wall_r_init, roof_r_init]))

        wall_c_init = torch.tensor([1.0 / (80.0 + i * 10.0) for i in range(n_wall)], dtype=torch.float32)
        roof_c_init = torch.tensor([1.0 / (60.0 + i * 10.0) for i in range(n_roof)], dtype=torch.float32)
        self.rc_C_inv = nn.Parameter(torch.cat([wall_c_init, roof_c_init]))

        if n_window > 0:
            window_r_init = torch.tensor([1.0 / (1.0 + i * 0.1) for i in range(n_window)], dtype=torch.float32)
            self.window_R_inv = nn.Parameter(window_r_init)

    def forward(self, T_amb, Tmid_list, Tzone, solar_gains_wall, solar_gains_roof):
        """
        Envelope heat transfer calculation

        Args:
            T_amb: Ambient temperature [batch, time, 1]
            Tmid_list: List of envelope temperatures [n_rc elements each [batch, time, 1]]
            Tzone: Zone temperature [batch, time, 1]
            solar_gains_wall: List of wall solar gains [n_wall elements each [batch, time, 1]]
            solar_gains_roof: List of roof solar gains [n_roof elements each [batch, time, 1]]

        Returns:
            dTmid_dt_list: List of RC temperature derivatives
            q_env_total: Total heat flow to zone
        """
        batch_size = T_amb.shape[0]

        # Stack all envelope temperatures into one tensor [batch, time, n_rc]
        Tmid_stacked = torch.cat(Tmid_list, dim=2)

        # Expand ambient and zone temperatures for broadcasting
        T_amb_expanded = T_amb.expand(-1, -1, self.n_rc)
        Tzone_expanded = Tzone.expand(-1, -1, self.n_rc)

        # Apply softplus to parameters and expand for broadcasting
        r_inv_bounded = F.softplus(self.rc_R_inv) * (1.0 / self.envelop["r_opaque_coef"])
        c_inv_bounded = F.softplus(self.rc_C_inv) * (1.0 / self.envelop["c_opaque_coef"])
        r_inv_expanded = r_inv_bounded.view(1, 1, -1).expand(batch_size, T_amb.shape[1], -1)
        c_inv_expanded = c_inv_bounded.view(1, 1, -1).expand(batch_size, T_amb.shape[1], -1)

        # Energy balance constraints
        q_in = (T_amb_expanded - Tmid_stacked) * r_inv_expanded
        q_out = (Tmid_stacked - Tzone_expanded) * r_inv_expanded

        # Combine solar gains into single tensor with same shape as Tmid_stacked
        # Add solar radiation to the first capacitance node
        solar_gains = []
        for i in range(self.n_wall):
            solar_gains.append(solar_gains_wall[i])
        for i in range(self.n_roof):
            solar_gains.append(solar_gains_roof[i])

        solar_stacked = torch.cat(solar_gains, dim=2) if solar_gains else torch.zeros_like(Tmid_stacked)

        # Calculate temperature derivatives
        dTmid_dt_stacked = (q_in - q_out + solar_stacked) * c_inv_expanded

        # Window heat transfer (Assume no storage capacity)
        q_window_total = 0
        if self.n_window > 0:
            window_r_inv_bounded = F.softplus(self.window_R_inv) * (1.0 / self.envelop["r_transparent_coef"])
            window_r_inv_expanded = window_r_inv_bounded.view(1, 1, -1).expand(batch_size, T_amb.shape[1], -1)
            T_amb_window = T_amb.expand(-1, -1, self.n_window)
            Tzone_window = Tzone.expand(-1, -1, self.n_window)
            q_window = (T_amb_window - Tzone_window) * window_r_inv_expanded
            q_window_total = torch.sum(q_window, dim=2, keepdim=True)

        # Total flow calculation
        q_rc_total = torch.sum(q_out, dim=2, keepdim=True)
        q_env_total = q_rc_total + q_window_total

        # Split to each state node
        dTmid_dt_list = torch.split(dTmid_dt_stacked, 1, dim=2)

        return dTmid_dt_list, q_env_total


class ModNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        para = args["para"]
        envelop = args["envelop"]
        self.enc_len = args["enLen"]
        self.device = args["device"]
        self.delta_t = 900  # 15 minutes in seconds

        # Model configuration
        self.n_wall = envelop.get("n_wall", 0)
        self.n_roof = envelop.get("n_roof", 0)
        self.n_window = envelop.get("n_window", 0)

        print(f"Initializing Vectorized ModNN with "
              f"{self.n_wall} walls, "
              f"{self.n_roof} roofs, "
              f"{self.n_window} windows")

        # Modules Assemble
        self.Int = Internal(para["Int_in"], para["Int_h"], para["Int_out"])
        self.HVAC = HVAC()
        self.solar = SolarRadiation(envelop=envelop)
        self.envelopes = EnvelopeRC(envelop, self.n_wall, self.n_roof, self.n_window)
        self.zone_update = ZoneUpdate(envelop=envelop)

    def forward(self, input_X):
        """
        input_X order: [T_zone, T_ambient, solar, day_sin, day_cos, occ, phvac]
        Shape: [batch_size, time_steps, features]
        """
        batch_size = input_X.shape[0]
        time_steps = input_X.shape[1]

        # Extract input data
        T_amb = input_X[:, :, [1]]  # Ambient air temperature
        Solar_X = input_X[:, :, [2]]  # Solar radiation
        Int_X = input_X[:, :, [3, 4, 5]]  # Time_sin/cos, Occ
        HVAC_X = input_X[:, :, [6]]  # pHVAC
        Tzone_gt = input_X[:, :, [0]]  # Zone air temperature groundtruth

        # Pre-allocate results arrays
        Tzone_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_int_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_hvac_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_env_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_solar_direct_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)

        # Pre-compute internal and HVAC
        q_int_pred = self.Int(Int_X)
        q_hvac_pred = self.HVAC(HVAC_X)

        # Initialize zone temperature
        Tzone = Tzone_gt[:, 0:1, :]

        # Initialize envelope temperatures
        Tmid = []
        for _ in range(self.n_wall + self.n_roof):
            tmid_init = 0.7 * Tzone + 0.3 * T_amb[:, 0:1, :]
            Tmid.append(tmid_init)

        # Pre-allocate envelope temperature histories
        Tmid_hist = [torch.zeros(batch_size, time_steps, 1, device=input_X.device)
                     for _ in range(self.n_wall + self.n_roof)]

        # --- Encoder phase ---
        for i in range(self.enc_len):
            # The purpose is to stablize the initial envelop temperatures so we use ground truth temperature
            Tzone_pred[:, i:i + 1, :] = Tzone_gt[:, i:i + 1, :]
            Tzone_i = Tzone_gt[:, i:i + 1, :]

            # Get inputs for this timestep
            T_amb_i = T_amb[:, i:i + 1, :]
            Solar_i = Solar_X[:, i:i + 1, :]

            # Calculate solar distribution
            q_solar_direct, q_solar_wall, q_solar_roof = self.solar(
                Solar_i, self.n_wall, self.n_roof
            )
            q_solar_direct_pred[:, i:i + 1, :] = q_solar_direct

            # Envelope heat transfer calculation
            dTmid_dt_list, q_env_total = self.envelopes(
                T_amb_i, Tmid, Tzone_i, q_solar_wall, q_solar_roof
            )
            q_env_pred[:, i:i + 1, :] = q_env_total

            # Envelope temperatures update
            # Use RK4 if want higher accuracy
            for idx, dTmid_dt in enumerate(dTmid_dt_list):
                # Clamp derivatives for stability
                dTmid_dt_clamped = torch.clamp(dTmid_dt, -3.0 / self.delta_t, 3.0 / self.delta_t)
                delta = torch.clamp(self.delta_t * dTmid_dt_clamped, -3.0, 3.0)
                Tmid[idx] = Tmid[idx] + delta
                Tmid_hist[idx][:, i:i + 1, :] = Tmid[idx]

        # --- Decoder phase (this is the actual prediction stage) ---
        Tzone = Tzone_gt[:, self.enc_len - 1:self.enc_len, :]

        for i in range(time_steps - self.enc_len):
            idx = self.enc_len + i

            # Initialize
            T_amb_i = T_amb[:, idx:idx + 1, :]
            Solar_i = Solar_X[:, idx:idx + 1, :]
            q_int_i = q_int_pred[:, idx:idx + 1, :]
            q_hvac_i = q_hvac_pred[:, idx:idx + 1, :]

            # Calculate solar distribution
            q_solar_direct, q_solar_wall, q_solar_roof = self.solar(
                Solar_i, self.n_wall, self.n_roof
            )
            q_solar_direct_pred[:, idx:idx + 1, :] = q_solar_direct

            # Envelope heat transfer calculation
            dTmid_dt_list, q_env_total = self.envelopes(
                T_amb_i, Tmid, Tzone, q_solar_wall, q_solar_roof
            )
            q_env_pred[:, idx:idx + 1, :] = q_env_total

            # Envelope temperatures update
            for j, dTmid_dt in enumerate(dTmid_dt_list):
                # Clamp derivatives for stability
                dTmid_dt_clamped = torch.clamp(dTmid_dt, -3.0 / self.delta_t, 3.0 / self.delta_t)
                delta = torch.clamp(self.delta_t * dTmid_dt_clamped, -3.0, 3.0)
                Tmid[j] = Tmid[j] + delta
                Tmid_hist[j][:, idx:idx + 1, :] = Tmid[j]

            # Calculate zone temperature derivative
            dTzone_dt = self.zone_update(q_env_total, q_int_i, q_hvac_i, q_solar_direct)
            # TODO: consider to integrated with Neural ODE

            # Update zone temperature with stability limits
            dTzone_dt_clamped = torch.clamp(dTzone_dt, -2.0 / self.delta_t, 2.0 / self.delta_t)
            delta = torch.clamp(self.delta_t * dTzone_dt_clamped, -2.0, 2.0)
            Tzone = Tzone + delta
            Tzone_pred[:, idx:idx + 1, :] = Tzone

        return Tzone_pred, Tzone_gt, (q_int_pred, q_hvac_pred, q_env_pred, q_solar_direct_pred)


# --- Runge-Kutta Integration ---
def rk4_step(f, x, dt, max_delta=3.0):
    k1 = f(x)
    k1 = torch.clamp(k1, -max_delta / dt, max_delta / dt)

    k2 = f(x + 0.5 * dt * k1)
    k2 = torch.clamp(k2, -max_delta / dt, max_delta / dt)

    k3 = f(x + 0.5 * dt * k2)
    k3 = torch.clamp(k3, -max_delta / dt, max_delta / dt)

    k4 = f(x + dt * k3)
    k4 = torch.clamp(k4, -max_delta / dt, max_delta / dt)

    dx = (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x + torch.clamp(dx, -max_delta, max_delta)


# --- Internal Heat Gain Module ---
class Internal(nn.Module):
    """
    A simple MLP is used to predict internal heat gain (convection ONLY)
    #TODO: add a RNN/MLP with lookback window to consider internal radiation

    In general, the internal heat gain comes from Lighting, Occupant, Appliance, etc,.
    But they can be represented by a factor (alpha) multiply with a "schedule" (sch), for example:
    q_light = alpha_light * sch_light
    q_human = alpha_human * sch_human
    q_cooking = alpha_cooking * sch_cooking

    If detailed information is available, we can replace it by Physics Equations
    For example, 80~100W * Number_of_people
    Otherwise, we learn form periodic features
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.FC1 = nn.Linear(input_size - 1, hidden_size)
        self.FC2 = nn.Linear(hidden_size, output_size)
        self.gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Initialize weights
        nn.init.xavier_uniform_(self.FC1.weight, gain=0.1)
        nn.init.zeros_(self.FC1.bias)
        nn.init.xavier_uniform_(self.FC2.weight, gain=0.1)
        nn.init.zeros_(self.FC2.bias)

    def forward(self, x):
        x1 = F.relu(self.FC1(x[:, :, :-1]))
        sch = torch.sigmoid(self.FC2(x1)) + x[:, :, -1:]
        return F.softplus(self.gain) * 0.1 * sch


# --- HVAC Module ---
class HVAC(nn.Module):
    """
    A simplified linear module is used here for AIR SIDE SYSTEM ONLY
    Change it to any type of RNN or add look back window for RADIATION SYSTEM
    The input is pre-calculated pHVAC (thermal load)

    But if the raw data is HVAC energy, or supply air flow/temperature, fan speed, coil load, etc
    #TODO: Add More Types of HVAC Modules

    """

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        return (F.softplus(self.gain) * 0.1 * x)


# --- Zone Heat Balance Module ---
class ZoneUpdate(nn.Module):
    def __init__(self, envelop, c_init=10000.0):
        super().__init__()
        self.C_inv = nn.Parameter(torch.tensor(1.0 / c_init, dtype=torch.float32))
        self.envelop = envelop

    def forward(self, q_env_total, q_int, q_hvac, q_solar_direct=None):
        c_inv_bounded = F.softplus(self.C_inv) * (1.0 / self.envelop["c_zone"])

        # Add direct solar gain if provided
        if q_solar_direct is not None:
            return (q_env_total + q_int + q_hvac + q_solar_direct) * c_inv_bounded
        else:
            return (q_env_total + q_int + q_hvac) * c_inv_bounded