from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from ..common.networks import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicyWeights(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        network_width: int,
        network_depth: int,
    ):
        super(GaussianPolicyWeights, self).__init__()

        state_dim, weights_dim = input_dim

        self.linear_input = nn.Linear(state_dim, network_width)
        self.pipeline = []
        for i in range(network_depth - 1):
            self.pipeline.append(nn.Linear(network_width, network_width))
        self.pipeline = nn.ModuleList(self.pipeline)
        self.mean_linear: nn.Linear = nn.Linear(network_width, weights_dim)
        self.log_std_linear: nn.Linear = nn.Linear(network_width, weights_dim)

        self.apply(weights_init_)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        weights = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_prob = normal.log_prob(weights).sum(dim=-1)

        return weights, log_prob.unsqueeze(dim=-1), mean

    def to(self, device: torch.device) -> "GaussianPolicyWeights":
        return super(GaussianPolicyWeights, self).to(device)
