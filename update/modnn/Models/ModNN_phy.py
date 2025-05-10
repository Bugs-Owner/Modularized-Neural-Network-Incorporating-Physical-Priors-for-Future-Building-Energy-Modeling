import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Solar Radiation Module ---
class SolarRadiation(nn.Module):
    def __init__(self, envelop, name="Solar"):
        super().__init__()
        # Initialization
        self.envelop = envelop

        # Solar transmission coefficient
        self.direct_gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Solar absorption coefficient
        self.abs_wall = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.abs_roof = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.name = name

    def forward(self, radiation, n_wall, n_roof):
        """
        Calculate solar heat gains [transmission and absorption]

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
        direct_coef   = F.softplus(self.direct_gain) * self.envelop["direct_coef"]
        abs_wall_coef = F.softplus(self.abs_wall) * self.envelop["abs_wall_coef"]
        abs_roof_coef = F.softplus(self.abs_roof) * self.envelop["abs_roof_coef"]

        # Calculate direct solar gain to zone
        direct_gain = direct_coef * radiation
        # TODO: Distribute transmission solar radiation to each surfaces

        # Calculate absorption by walls
        wall_gains = []
        for i in range(n_wall):
            # Different walls can have different orientations
            # TODO: ASHRAE Clear-Sky Model
            if radiation.shape[-1] > 1 and i < radiation.shape[-1]:
                # Use specific direction for this wall
                wall_gain = abs_wall_coef * radiation[:, :, i % radiation.shape[-1]]
                wall_gains.append(wall_gain.unsqueeze(-1))
            else:
                # Use total radiation
                wall_gain = abs_wall_coef * radiation
                wall_gains.append(wall_gain)

        # Calculate absorption by roof (assume horizontal)
        roof_gains = []
        for i in range(n_roof):
            roof_gain = abs_roof_coef * radiation
            roof_gains.append(roof_gain)

        return direct_gain, wall_gains, roof_gains


# --- Envelope Module [RC] ---
class EnvelopeRC(nn.Module):
    def __init__(self, envelop, name="RC", r_init=10.0, c_init=100.0):
        super().__init__()
        self.R_inv = nn.Parameter(torch.tensor(1.0 / r_init, dtype=torch.float32))
        self.C_inv = nn.Parameter(torch.tensor(1.0 / c_init, dtype=torch.float32))
        self.name = name
        self.envelop = envelop

    def forward(self, Tamb, Tmid, Tzone, solar_gain=None):
        # Bounded parameters for stability
        r_inv_bounded = F.softplus(self.R_inv) * (1.0 / self.envelop["r_opaque_coef"])
        c_inv_bounded = F.softplus(self.C_inv) * (1.0 / self.envelop["c_opaque_coef"])

        # Calculate heat flows
        q_in = (Tamb - Tmid) * r_inv_bounded
        q_out = (Tmid - Tzone) * r_inv_bounded

        # Add solar gain if provided
        if solar_gain is not None:
            # Add solar radiation to the first capacitance node
            dTmid_dt = (q_in - q_out + solar_gain) * c_inv_bounded
        else:
            dTmid_dt = (q_in - q_out) * c_inv_bounded

        return dTmid_dt, q_out


# --- Envelope Module [R] ---
class EnvelopeROnly(nn.Module):
    def __init__(self, envelop, name="R-only", r_init=0.1):
        super().__init__()
        self.R_inv = nn.Parameter(torch.tensor(1/r_init, dtype=torch.float32))
        self.name = name
        self.envelop = envelop

    def forward(self, Tamb, Tzone):
        r_inv_bounded = F.softplus(self.R_inv) * (1.0 / self.envelop["r_transparent_coef"])

        return (Tamb - Tzone) * r_inv_bounded


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


class ZoneUpdate(nn.Module):
    def __init__(self, envelop, c_init=10000.0):
        super().__init__()
        self.C_inv = nn.Parameter(torch.tensor(1.0 / c_init, dtype=torch.float32))
        self.envelop = envelop

    def forward(self, q_env_total, q_int, q_hvac, q_solar_direct=None):
        # Bounded parameter
        c_inv_bounded = F.softplus(self.C_inv) * (1.0 / self.envelop["c_zone"])

        # Add direct solar gain if provided
        if q_solar_direct is not None:
            return (q_env_total + q_int + q_hvac + q_solar_direct) * c_inv_bounded
        else:
            return (q_env_total + q_int + q_hvac) * c_inv_bounded


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

        print(f"Initializing ModNN with {self.n_wall} walls, {self.n_roof} roofs, {self.n_window} windows")

        # Assemble modules
        # Internal heat gain module
        self.Int = Internal(para["Int_in"], para["Int_h"], para["Int_out"])
        # HVAC module
        self.HVAC = HVAC()
        # Envelope module
        self.solar = SolarRadiation(envelop=envelop)
        self.rc_envelopes = nn.ModuleList([
                                              EnvelopeRC(envelop=envelop,
                                                         name=f"wall_{i}",
                                                         r_init=10.0 + i * 1.0,
                                                         c_init=80.0 + i * 10.0)
                                              for i in range(self.n_wall)
                                          ] + [
                                              EnvelopeRC(envelop=envelop,
                                                         name=f"roof_{i}",
                                                         r_init=20.0 + i * 2.0,
                                                         c_init=60.0 + i * 10.0)
                                              for i in range(self.n_roof)
                                          ])

        self.r_only_envelopes = nn.ModuleList([
            EnvelopeROnly(envelop=envelop,
                          name=f"window_{i}",
                          r_init=1 + i * 0.1)
            for i in range(self.n_window)
        ])

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

        # Pre-allocate results arrays - more efficient than appending
        Tzone_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_int_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_hvac_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_env_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)
        q_solar_direct_pred = torch.zeros(batch_size, time_steps, 1, device=input_X.device)

        # Pre-allocate envelope temperature histories
        Tmid_hist = [torch.zeros(batch_size, time_steps, 1, device=input_X.device)
                     for _ in range(len(self.rc_envelopes))]

        # Initialize zone temperature for first step
        Tzone = Tzone_gt[:, 0:1, :]

        # Initialize envelope temperatures (only once)
        Tmid = []
        for _ in range(len(self.rc_envelopes)):
            tmid_init = 0.7 * Tzone + 0.3 * T_amb[:, 0:1, :]
            Tmid.append(tmid_init)

        # --- Encoder phase (optimization: only calculate what's needed) ---
        for i in range(self.enc_len):
            # Simply record the ground truth temperature
            Tzone_pred[:, i:i + 1, :] = Tzone_gt[:, i:i + 1, :]

            # For envelope temperatures, we need to calculate and track these
            # even during encoder phase as they're not directly observed
            T_amb_i = T_amb[:, i:i + 1, :]
            Solar_i = Solar_X[:, i:i + 1, :]
            Tzone_i = Tzone_gt[:, i:i + 1, :]

            # Calculate solar distribution (needed for envelope temperatures)
            q_solar_direct, q_solar_wall, q_solar_roof = self.solar(
                Solar_i, self.n_wall, self.n_roof
            )
            q_solar_direct_pred[:, i:i + 1, :] = q_solar_direct

            # Update envelope temperatures
            for idx, env in enumerate(self.rc_envelopes):
                # Assign solar gains
                if idx < self.n_wall and q_solar_wall:
                    solar_gain = q_solar_wall[idx]
                elif idx >= self.n_wall and q_solar_roof:
                    solar_gain = q_solar_roof[idx - self.n_wall]
                else:
                    solar_gain = None

                # Evolution function for rk4
                def dTmid(Tmid_local):
                    dTdt, _ = env(T_amb_i, Tmid_local, Tzone_i, solar_gain)
                    return dTdt

                # Update envelope temperature using RK4
                Tmid[idx] = rk4_step(dTmid, Tmid[idx], self.delta_t)
                Tmid_hist[idx][:, i:i + 1, :] = Tmid[idx]

            # Only calculate these if needed for analysis, otherwise leave as zeros
            # in encoder phase since they don't affect the state evolution
            Int_i = Int_X[:, i:i + 1, :]
            HVAC_i = HVAC_X[:, i:i + 1, :]
            q_int_pred[:, i:i + 1, :] = self.Int(Int_i)
            q_hvac_pred[:, i:i + 1, :] = self.HVAC(HVAC_i)

            # Optional: Calculate envelope heat transfer contributions
            # if needed for analysis. If not, can skip this.
            q_env_total = torch.zeros_like(Tzone)
            for idx, env in enumerate(self.rc_envelopes):
                if idx < self.n_wall and q_solar_wall:
                    solar_gain = q_solar_wall[idx]
                elif idx >= self.n_wall and q_solar_roof:
                    solar_gain = q_solar_roof[idx - self.n_wall]
                else:
                    solar_gain = None
                _, q_env = env(T_amb_i, Tmid[idx], Tzone_i, solar_gain)
                q_env_total += q_env

            for env in self.r_only_envelopes:
                q_env_r = env(T_amb_i, Tzone_i)
                q_env_total += q_env_r

            q_env_pred[:, i:i + 1, :] = q_env_total

        # --- Decoder phase (predicting future temps) ---
        # Starting from the last encoder step
        Tzone = Tzone_gt[:, self.enc_len - 1:self.enc_len, :]

        for i in range(time_steps - self.enc_len):
            idx = self.enc_len + i
            T_amb_i = T_amb[:, idx:idx + 1, :]
            Int_i = Int_X[:, idx:idx + 1, :]
            HVAC_i = HVAC_X[:, idx:idx + 1, :]
            Solar_i = Solar_X[:, idx:idx + 1, :]

            # Process internal gains and HVAC - only calculate once
            q_int = self.Int(Int_i)
            q_hvac = self.HVAC(HVAC_i)
            q_int_pred[:, idx:idx + 1, :] = q_int
            q_hvac_pred[:, idx:idx + 1, :] = q_hvac

            # Distribute solar radiation
            q_solar_direct, q_solar_wall, q_solar_roof = self.solar(
                Solar_i, self.n_wall, self.n_roof
            )
            q_solar_direct_pred[:, idx:idx + 1, :] = q_solar_direct

            # Prepare for zone temperature update
            q_env_total = torch.zeros_like(Tzone)

            # Update envelope temperatures and calculate heat flow
            for j, env in enumerate(self.rc_envelopes):
                # Assign solar gains
                if j < self.n_wall and q_solar_wall:
                    solar_gain = q_solar_wall[j]
                elif j >= self.n_wall and q_solar_roof:
                    solar_gain = q_solar_roof[j - self.n_wall]
                else:
                    solar_gain = None

                def dTmid(Tmid_local):
                    dTdt, _ = env(T_amb_i, Tmid_local, Tzone, solar_gain)
                    return dTdt

                # Update envelope temperature
                Tmid[j] = rk4_step(dTmid, Tmid[j], self.delta_t)
                Tmid_hist[j][:, idx:idx + 1, :] = Tmid[j]

                # Calculate heat flow contribution
                _, q_env = env(T_amb_i, Tmid[j], Tzone, solar_gain)
                q_env_total += q_env

            # Add window heat flow
            for env in self.r_only_envelopes:
                q_env_r = env(T_amb_i, Tzone)
                q_env_total += q_env_r

            q_env_pred[:, idx:idx + 1, :] = q_env_total

            # Update zone temperature using RK4
            def dTzone(Tzone_local):
                return self.zone_update(q_env_total, q_int, q_hvac, q_solar_direct)

            # Update zone temperature with stability limits
            Tzone = rk4_step(dTzone, Tzone, self.delta_t, max_delta=2.0)
            Tzone_pred[:, idx:idx + 1, :] = Tzone
        #TODO: the second term should be envelop temperature, but we need to revise "Play" function, leave it later
        return Tzone_pred, Tzone_gt, (q_int_pred, q_hvac_pred, q_env_pred, q_solar_direct_pred)