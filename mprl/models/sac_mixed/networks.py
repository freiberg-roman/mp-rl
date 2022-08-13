from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from mprl.models.sac_common.networks import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianMotionPrimitivePolicy(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_weights: int,
        hidden_dim: int,
    ):
        super(GaussianMotionPrimitivePolicy, self).__init__()
        self.num_weights: int = num_weights

        self.linear1: nn.Linear = nn.Linear(num_inputs, hidden_dim)
        self.linear2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear: nn.Linear = nn.Linear(hidden_dim, num_weights)

        self.log_std_linear: nn.Linear = nn.Linear(hidden_dim, num_weights)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.linear1(state))
        x = F.silu(self.linear2(x))
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

        return (
            weights,
            log_prob.unsqueeze(dim=-1),
            mean,
            {"std": std, "mean": mean},
        )  # used for logging

    def to(self, device: torch.device) -> "GaussianMotionPrimitivePolicy":
        return super(GaussianMotionPrimitivePolicy, self).to(device)
