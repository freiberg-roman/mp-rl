from typing import Tuple

import numpy as np
import torch
import torch as ch
from torch.distributions import Normal

from mprl.trd_party.trl.trust_region_projections.models.policy.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from mprl.trd_party.trl.trust_region_projections.projections.kl_projection_layer import (
    KLProjectionLayer,
)
from mprl.trd_party.trl.trust_region_projections.projections.w2_projection_layer import (
    WassersteinProjectionLayer,
)

from ...utils.math_helper import hard_update, soft_update
from ..sac.networks import GaussianPolicy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianPolicyStub(AbstractGaussianPolicy):
    def __init__(self):
        self.contextual_std = True

    def _get_std_parameter(self, action_dim):
        raise NotImplemented

    def _get_std_layer(self, prev_size, action_dim, init):
        raise NotImplemented

    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        raise NotImplemented

    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        raise NotImplemented

    def log_probability(
        self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs
    ) -> ch.Tensor:
        raise NotImplemented

    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]) -> ch.Tensor:
        _, std = p
        logdet = self.log_determinant(std)
        k = std.shape[-1]
        return 0.5 * (k * np.log(2 * np.e * np.pi) + logdet)

    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        std = std.diagonal(dim1=-2, dim2=-1)
        return 2 * std.log().sum(-1)

    def maha(self, mean, mean_other, std) -> ch.Tensor:
        std = std.diagonal(dim1=-2, dim2=-1)
        diff = mean - mean_other
        return (diff / std).pow(2).sum(-1)

    def precision(self, std: ch.Tensor) -> ch.Tensor:
        return 1 / self.covariance(std)

    def covariance(self, std) -> ch.Tensor:
        return std.pow(2)

    def set_std(self, std: ch.Tensor) -> None:
        raise NotImplemented

    def forward(self, x, train=True):
        raise NotImplemented

    def is_diag(self):
        return True


class TrustRegionPolicy:
    def __init__(
        self,
        input_dim: Tuple[int, int],
        network_width: int,
        network_depth: int,
        action_scale: float = 1.0,
        eps: float = 0.03,
        eps_cov: float = 0.001,
    ):
        self.policy = GaussianPolicy(
            input_dim, network_width, network_depth, action_scale
        )
        self.old_policy = GaussianPolicy(
            input_dim, network_width, network_depth, action_scale
        )
        self.hard_update()
        self.trl = WassersteinProjectionLayer("w2", mean_bound=eps, cov_bound=eps_cov)
        self.policy_stub = GaussianPolicyStub()

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy.forward(state)
        std = log_std.exp()
        old_mean, old_log_std = self.old_policy.forward(state)
        old_std = old_log_std.exp()
        if self.trl.initial_entropy is None:
            self.trl.initial_entropy = self.policy_stub.entropy((mean, std)).mean()

        (mean_proj, std_proj) = self.trl(
            self.policy_stub,
            (mean, std.diag_embed()),
            (old_mean, old_std.diag_embed()),
            0,
        )
        std_proj = std_proj.diagonal(dim1=-2, dim2=-1)
        normal = Normal(mean_proj, std_proj)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.policy.action_scale + self.policy.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.policy.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_after_proj = (
            torch.tanh(mean_proj) * self.policy.action_scale + self.policy.action_bias
        )
        return action, log_prob, mean_after_proj, (mean, std), (mean_proj, std_proj)

    def hard_update(self):
        """Called after each update step"""
        hard_update(self.old_policy, self.policy)

    def soft_update(self, tau):
        soft_update(self.old_policy, self.policy, tau)

    def kl_regularization_loss(self, p, proj_p):
        p_mean, p_std = p
        p_proj_mean, p_proj_std = proj_p
        return self.trl.get_trust_region_loss(
            self.policy_stub,
            (p_mean, p_std.diag_embed()),
            (p_proj_mean, p_proj_std.diag_embed()),
        )

    def to(self, device: torch.device) -> "TrustRegionPolicy":
        self.policy.to(device)
        self.old_policy.to(device)
        return self

    def parameters(self):
        return self.policy.parameters()
