import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        para = args["para"]
        self.device = args["device"]

        self.horizon = 96
        input_dim = para["current_meas_dim"] + para["future_disturbance_dim"] * self.horizon
        hidden_dim = para["hidden_dim"]

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.horizon),
            nn.Tanh()
        )

    def forward(self, obs):
        """
        obs: [batch, 144, 10]
        "0.T_zone, 1.T_ambient, 2.solar, 3.day_sin, 4.day_cos,
        5.occ, 6.phvac, 7.setpt_cool, 8.setpt_heat, 9.price"
        Returns: [batch, 96] predicted control trajectory
        """

        # 1. Current measurements at t=47 (T_zone + 6 others)
        current = obs[:, 47, [0, 1, 2, 5, 7, 8, 9]]  # [batch, 7]

        # 2. Future known disturbances (exclude T_zone and phvac)
        future = obs[:, 48:, [1, 2, 5, 7, 8, 9]]  # [batch, 96, 6]
        future = future.reshape(obs.shape[0], -1)  # [batch, 768]

        # Combine
        x_input = torch.cat([current, future], dim=1)  # [batch, 775]

        return self.model(x_input).unsqueeze(-1)
