from typing import Tuple

import torch as ch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform

from mprl.models.common import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        network_width: int,
        network_depth: int,
        action_scale: float = 1.0,
    ):
        super(GaussianPolicy, self).__init__()

        state_dim, action_dim = input_dim
        self.linear_input: nn.Linear = nn.Linear(state_dim, network_width)
        self.pipeline = []
        for _ in range(network_depth - 1):
            self.pipeline.append(nn.Linear(network_width, network_width))
        self.pipeline = nn.ModuleList(self.pipeline)
        self.mean_linear: nn.Linear = nn.Linear(network_width, action_dim)
        self.log_std_linear: nn.Linear = nn.Linear(network_width, action_dim)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale: ch.Tensor = ch.tensor(action_scale)
        self.action_bias: ch.Tensor = ch.tensor(0.0)

    def forward(self, state: ch.Tensor) -> Tuple[ch.Tensor, ch.Tensor]:
        x = F.silu(self.linear_input(state))
        for layer in self.pipeline:
            x = F.silu(layer(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = ch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    @ch.no_grad()
    def sample(self, state: ch.Tensor) -> Tuple[ch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = ch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action

    @ch.no_grad()
    def sample_no_tanh(self, state: ch.Tensor) -> Tuple[ch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        return normal.rsample()

    def sample_log_prob(self, state: ch.Tensor) -> Tuple[ch.Tensor, ch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = ch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= ch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    @ch.no_grad()
    def sample_log_prob_no_tanh(self, state: ch.Tensor) -> Tuple[ch.Tensor, ch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(1, keepdim=True)
        return action, log_prob

    @ch.no_grad()
    def mean(self, state: ch.Tensor) -> ch.Tensor:
        mean, _ = self.forward(state)
        mean = ch.tanh(mean) * self.action_scale + self.action_bias
        return mean

    @ch.no_grad()
    def mean_no_tanh(self, state: ch.Tensor) -> ch.Tensor:
        mean, _ = self.forward(state)
        return mean
