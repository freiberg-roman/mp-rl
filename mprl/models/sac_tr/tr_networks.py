from typing import Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from mprl.trd_party.trl.trust_region_projections.models.policy.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from mprl.trd_party.trl.trust_region_projections.projections.kl_projection_layer import (
    KLProjectionLayer,
)

from ...utils.math_helper import hard_update
from ..sac.networks import GaussianPolicy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicyStub(AbstractGaussianPolicy):
    def is_diag(self):
        return True


class TrustRegionPolicy:
    def __init__(
        self,
        input_dim: Tuple[int, int],
        network_width: int,
        network_depth: int,
        action_scale: float = 1.0,
        eps: float = 0.001,
        eps_cov: float = 0.001,
    ):
        self.policy = GaussianPolicy(
            input_dim, network_width, network_depth, action_scale
        )
        self.old_policy = GaussianPolicy(
            input_dim, network_width, network_depth, action_scale
        )
        self.hard_update()
        self.trl = KLProjectionLayer("kl")
        self.policy_stub = GaussianPolicyStub()
        self.eps = eps
        self.eps_cov = eps_cov

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.linear_input(state))
        for layer in self.pipeline:
            x = F.silu(layer(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy.sample(state)
        std = log_std.exp()
        old_mean, old_log_std = self.old_policy(state)
        old_std = old_log_std.exp()
        (mean_proj, std_proj) = self.trl(
            self.policy_stub, (mean, std), (old_mean, old_std), self.eps, self.eps_cov
        )

        normal = Normal(mean_proj, std_proj)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_after_proj = torch.tanh(mean_proj) * self.action_scale + self.action_bias
        return action, log_prob, mean_after_proj

    def hard_update(self):
        """Called after each update step"""
        hard_update(self.old_policy, self.policy)

    def to(self, device: torch.device) -> "TrustRegionPolicy":
        self.policy.to(device)
        self.old_policy.to(device)
        return self

    def parameters(self):
        return self.policy.parameters()
