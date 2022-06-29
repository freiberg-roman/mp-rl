import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Gamma, Normal

from mprl.models.sac_common.networks import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianMotionPrimitiveTimePolicy(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_weights: int,
        hidden_dim: int,
        min_time: float = 0.05,
        max_time: float = 1.0,
    ):
        super(GaussianMotionPrimitiveTimePolicy, self).__init__()
        self.num_weights = num_weights
        self.min_time = min_time
        self.max_time = max_time

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_weights)

        self.time_alpha_linear = nn.Linear(hidden_dim, 1)
        self.time_beta_linear = nn.Linear(hidden_dim, 1)
        self.scalar_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scalar_beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.sp = nn.Softplus()
        self.sig = nn.Sigmoid()

        self.log_std_linear = nn.Linear(hidden_dim, num_weights)

        self.apply(weights_init_)
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        x = F.silu(self.linear1(state))
        x = F.silu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        lin_out_alpha = self.time_alpha_linear(x)
        lin_out_beta = self.time_beta_linear(x)
        gamma_alpha = self.sp(self.scalar_alpha * lin_out_alpha)
        gamma_beta = self.sp(self.scalar_beta * lin_out_beta)
        gamma_alpha = torch.clamp(gamma_alpha, min=self.min_time, max=self.max_time / 2)
        gamma_beta = torch.clamp(gamma_beta, min=self.min_time, max=self.max_time / 2)

        return mean, log_std, (gamma_alpha, gamma_beta)

    def sample(self, state):
        mean, log_std, t = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        weights = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        log_prob = normal.log_prob(weights)

        # tanh transformation
        weights = torch.tanh(weights)
        weights = weights * self.action_scale + self.action_bias
        log_prob -= torch.log(self.action_scale * (1 - weights.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1)

        # reparametrization

        gamma_dist = Gamma(t[0], t[1])  # time is sampled from a gamma distribution
        time = torch.clamp(
            gamma_dist.rsample(), min=self.min_time, max=self.max_time * 2
        )
        log_prob += torch.sum(gamma_dist.log_prob(time), dim=-1)
        return (
            torch.cat([weights, time], 1),
            log_prob.unsqueeze(dim=-1),
            torch.cat([mean, t[0] * t[1]], 1).detach(),
            {},  # used for logging
        )

    def to(self, device):
        return super(GaussianMotionPrimitiveTimePolicy, self).to(device)
