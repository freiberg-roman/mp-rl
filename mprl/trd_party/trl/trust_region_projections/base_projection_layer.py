#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple, Union

import torch as ch

from mprl.trd_party.trl.trust_region_projections.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from mprl.trd_party.trl.trust_region_projections.utils.lib import tensorize
from mprl.trd_party.trl.trust_region_projections.utils.projection import (
    gaussian_kl,
    get_entropy_schedule,
)


def entropy_inequality_projection(
    policy: AbstractGaussianPolicy,
    p: Tuple[ch.Tensor, ch.Tensor],
    beta: Union[float, ch.Tensor],
):
    mean, std = p
    k = std.shape[-1]
    batch_shape = std.shape[:-2]

    ent = policy.entropy(p)
    mask = ent < beta

    # if nothing has to be projected skip computation
    if (~mask).all():
        return p

    alpha = ch.ones(batch_shape, dtype=std.dtype, device=std.device)
    alpha[mask] = ch.exp((beta[mask] - ent[mask]) / k)

    proj_std = ch.einsum("ijk,i->ijk", std, alpha)
    return mean, ch.where(mask[..., None, None], proj_std, std)


def entropy_equality_projection(
    policy: AbstractGaussianPolicy,
    p: Tuple[ch.Tensor, ch.Tensor],
    beta: Union[float, ch.Tensor],
):
    mean, std = p
    k = std.shape[-1]

    ent = policy.entropy(p)
    alpha = ch.exp((beta - ent) / k)
    proj_std = ch.einsum("ijk,i->ijk", std, alpha)
    return mean, proj_std


def mean_projection(
    mean: ch.Tensor, old_mean: ch.Tensor, maha: ch.Tensor, eps: ch.Tensor
):
    batch_shape = mean.shape[:-1]
    mask = maha > eps

    # if nothing has to be projected skip computation
    if mask.any():
        omega = ch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = ch.sqrt(maha[mask] / eps) - 1.0
        omega = ch.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = ch.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean


class BaseProjectionLayer(object):
    def __init__(
        self,
        proj_type: str = "",
        mean_bound: float = 0.03,
        cov_bound: float = 1e-3,
        trust_region_coeff: float = 0.0,
        scale_prec: bool = True,
        entropy_schedule: Union[None, str] = None,
        action_dim: Union[None, int] = None,
        total_train_steps: Union[None, int] = None,
        target_entropy: float = 0.0,
        temperature: float = 0.5,
        entropy_eq: bool = False,
        entropy_first: bool = False,
        do_regression: bool = False,
        regression_iters: int = 1000,
        regression_lr: int = 3e-4,
        optimizer_type_reg: str = "adam",
        cpu: bool = True,
        dtype: ch.dtype = ch.float32,
    ):
        # projection and bounds
        self.proj_type = proj_type
        self.mean_bound = tensorize(mean_bound, cpu=cpu, dtype=dtype)
        self.cov_bound = tensorize(cov_bound, cpu=cpu, dtype=dtype)
        self.trust_region_coeff = trust_region_coeff
        self.scale_prec = scale_prec

        # projection utils
        assert (action_dim and total_train_steps) if entropy_schedule else True
        self.entropy_proj = (
            entropy_equality_projection if entropy_eq else entropy_inequality_projection
        )
        self.entropy_schedule = get_entropy_schedule(
            entropy_schedule, total_train_steps, dim=action_dim
        )
        self.target_entropy = tensorize(target_entropy, cpu=cpu, dtype=dtype)
        self.entropy_first = entropy_first
        self.entropy_eq = entropy_eq
        self.temperature = temperature
        self._initial_entropy = None

        # regression
        self.do_regression = do_regression
        self.regression_iters = regression_iters
        self.lr_reg = regression_lr
        self.optimizer_type_reg = optimizer_type_reg

    def __call__(
        self, policy, p: Tuple[ch.Tensor, ch.Tensor], q, step, *args, **kwargs
    ):
        # entropy_bound = self.policy.entropy(q) - self.target_entropy
        entropy_bound = self.entropy_schedule(
            self.initial_entropy, self.target_entropy, self.temperature, step
        ) * p[0].new_ones(p[0].shape[0])
        return self._projection(
            policy, p, q, self.mean_bound, self.cov_bound, entropy_bound, **kwargs
        )

    def _trust_region_projection(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        q: Tuple[ch.Tensor, ch.Tensor],
        eps: ch.Tensor,
        eps_cov: ch.Tensor,
        **kwargs
    ):
        return p

    # @final
    def _projection(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        q: Tuple[ch.Tensor, ch.Tensor],
        eps: ch.Tensor,
        eps_cov: ch.Tensor,
        beta: ch.Tensor,
        **kwargs
    ):
        # entropy projection in the beginning
        if self.entropy_first:
            p = self.entropy_proj(policy, p, beta)
        # trust region projection for mean and cov bounds
        proj_mean, proj_std = self._trust_region_projection(
            policy, p, q, eps, eps_cov, **kwargs
        )
        # entropy projection in the end
        if self.entropy_first:
            return proj_mean, proj_std

        return self.entropy_proj(policy, (proj_mean, proj_std), beta)

    @property
    def initial_entropy(self):
        return self._initial_entropy

    @initial_entropy.setter
    def initial_entropy(self, entropy):
        if self.initial_entropy is None:
            self._initial_entropy = entropy

    def trust_region_value(self, policy, p, q):
        return gaussian_kl(policy, p, q)

    def get_trust_region_loss(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        proj_p: Tuple[ch.Tensor, ch.Tensor],
    ):
        """
        Compute the trust region loss to ensure policy output and projection stay close.
        Args:
            policy: policy instance
            proj_p: projected distribution
            p: predicted distribution from network output

        Returns:
            trust region loss
        """
        p_target = (proj_p[0].detach(), proj_p[1].detach())
        mean_diff, cov_diff = self.trust_region_value(policy, p, p_target)

        delta_loss = (
            mean_diff + cov_diff if policy.contextual_std else mean_diff
        ).mean()

        return delta_loss

    def compute_metrics(self, policy, p, q) -> dict:
        """
        Returns dict with constraint metrics.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution

        Returns:
            dict with constraint metrics
        """
        with ch.no_grad():
            entropy_old = policy.entropy(q)
            entropy = policy.entropy(p)
            mean_kl, cov_kl = gaussian_kl(policy, p, q)
            kl = mean_kl + cov_kl

            mean_diff, cov_diff = self.trust_region_value(policy, p, q)

            combined_constraint = mean_diff + cov_diff
            entropy_diff = entropy_old - entropy

        return {
            "kl": kl.detach().mean(),
            "constraint": combined_constraint.mean(),
            "mean_constraint": mean_diff.mean(),
            "cov_constraint": cov_diff.mean(),
            "entropy": entropy.mean(),
            "entropy_diff": entropy_diff.mean(),
            "kl_max": kl.max(),
            "constraint_max": combined_constraint.max(),
            "mean_constraint_max": mean_diff.max(),
            "cov_constraint_max": cov_diff.max(),
            "entropy_max": entropy.max(),
            "entropy_diff_max": entropy_diff.max(),
        }
