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

import numpy as np
import torch as ch

from mprl.trd_party.trl.trust_region_projections.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from mprl.trd_party.trl.trust_region_projections.utils.lib import torch_batched_trace


def mean_distance(policy, mean, mean_other, std_other=None, scale_prec=False):
    if scale_prec:
        # maha objective for mean
        mean_part = policy.maha(mean, mean_other, std_other)
    else:
        # euclidean distance for mean
        # mean_part = ch.norm(mean_other - mean, ord=2, axis=1) ** 2
        mean_part = ((mean_other - mean) ** 2).sum(1)

    return mean_part


def gaussian_kl(
    policy: AbstractGaussianPolicy,
    p: Tuple[ch.Tensor, ch.Tensor],
    q: Tuple[ch.Tensor, ch.Tensor],
) -> Tuple[ch.Tensor, ch.Tensor]:
    mean, std = p
    mean_other, std_other = q
    k = mean.shape[-1]

    det_term = policy.log_determinant(std)
    det_term_other = policy.log_determinant(std_other)

    cov = policy.covariance(std)
    prec_other = policy.precision(std_other)

    maha_part = 0.5 * policy.maha(mean, mean_other, std_other)
    # trace_part = (var * precision_other).sum([-1, -2])
    trace_part = torch_batched_trace(prec_other @ cov)
    cov_part = 0.5 * (trace_part - k + det_term_other - det_term)

    return maha_part, cov_part


def gaussian_wasserstein_commutative(
    policy: AbstractGaussianPolicy,
    p: Tuple[ch.Tensor, ch.Tensor],
    q: Tuple[ch.Tensor, ch.Tensor],
    scale_prec=False,
) -> Tuple[ch.Tensor, ch.Tensor]:
    mean, sqrt = p
    mean_other, sqrt_other = q

    mean_part = mean_distance(policy, mean, mean_other, sqrt_other, scale_prec)

    cov = policy.covariance(sqrt)
    if scale_prec:
        # cov constraint scaled with precision of old dist
        batch_dim, dim = mean.shape

        identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = ch.linalg.solve(sqrt_other, identity)[0]
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        cov_part = torch_batched_trace(identity + c - 2 * sqrt_inv_other @ sqrt)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = policy.covariance(sqrt_other)
        cov_part = torch_batched_trace(cov_other + cov - 2 * sqrt_other @ sqrt)

    return mean_part, cov_part


def gaussian_wasserstein_non_commutative(
    policy: AbstractGaussianPolicy,
    p: Tuple[ch.Tensor, ch.Tensor],
    q: Tuple[ch.Tensor, ch.Tensor],
    scale_prec=False,
    return_eig=False,
) -> Union[
    Tuple[ch.Tensor, ch.Tensor], Tuple[ch.Tensor, ch.Tensor, ch.Tensor, ch.Tensor]
]:
    mean, sqrt = p
    mean_other, sqrt_other = q
    batch_dim, dim = mean.shape

    mean_part = mean_distance(policy, mean, mean_other, sqrt_other, scale_prec)

    cov = policy.covariance(sqrt)

    if scale_prec:
        # cov constraint scaled with precision of old dist
        # W2 objective for cov assuming normal W2 objective for mean
        identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = ch.solve(identity, sqrt_other)[0]
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        # compute inner parenthesis of trace in W2,
        eigvals, eigvecs = ch.symeig(c, eigenvectors=return_eig, upper=False)
        cov_part = torch_batched_trace(identity + c) - 2 * eigvals.sqrt().sum(1)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = policy.covariance(sqrt_other)

        # compute inner parenthesis of trace in W2,
        eigvals, eigvecs = ch.symeig(
            cov @ cov_other, eigenvectors=return_eig, upper=False
        )
        cov_part = torch_batched_trace(cov_other + cov) - 2 * eigvals.sqrt().sum(1)

    if return_eig:
        return mean_part, cov_part, eigvals, eigvecs

    return mean_part, cov_part


def get_entropy_schedule(schedule_type, total_train_steps, dim):
    if schedule_type == "linear":
        return (
            lambda initial_entropy, target_entropy, temperature, step: step
            * (target_entropy - initial_entropy)
            / total_train_steps
            + initial_entropy
        )
    elif schedule_type == "exp":
        return (
            lambda initial_entropy, target_entropy, temperature, step: dim
            * target_entropy
            + (initial_entropy - dim * target_entropy)
            * temperature ** (10 * step / total_train_steps)
        )
    else:
        return lambda initial_entropy, target_entropy, temperature, step: initial_entropy.new(
            [-np.inf]
        )
