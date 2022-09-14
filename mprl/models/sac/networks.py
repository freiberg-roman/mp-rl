from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..common import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(
        self, input_dim: Tuple[int, int], network_width: int, network_depth: int
    ):
        super(GaussianPolicy, self).__init__()

        state_dim, action_dim = input_dim
        self.linear_input: nn.Linear = nn.Linear(state_dim, network_width)
        self.pipeline = []
        for i in range(network_depth - 1):
            self.pipeline.append(nn.Linear(network_width, network_width))

        self.mean_linear: nn.Linear = nn.Linear(network_width, action_dim)
        self.log_std_linear: nn.Linear = nn.Linear(network_width, action_dim)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale: torch.Tensor = torch.tensor(1.0)
        self.action_bias: torch.Tensor = torch.tensor(0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.linear_input(state))
        for l in self.pipeline:
            x = F.silu(l(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device: torch.device) -> "GaussianPolicy":
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
