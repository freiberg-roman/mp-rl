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

from typing import Tuple

import torch as ch

from mprl.trd_party.trl.trust_region_projections.abstract_gaussian_policy import (
    AbstractGaussianPolicy,
)
from mprl.trd_party.trl.trust_region_projections.base_projection_layer import (
    BaseProjectionLayer,
    mean_projection,
)
from mprl.trd_party.trl.trust_region_projections.utils.projection import (
    gaussian_wasserstein_commutative,
)


class WassersteinProjectionLayer(BaseProjectionLayer):
    def _trust_region_projection(
        self,
        policy: AbstractGaussianPolicy,
        p: Tuple[ch.Tensor, ch.Tensor],
        q: Tuple[ch.Tensor, ch.Tensor],
        eps: ch.Tensor,
        eps_cov: ch.Tensor,
        **kwargs
    ):
        mean, sqrt = p
        old_mean, old_sqrt = q
        batch_shape = mean.shape[:-1]
        mean_part, cov_part = gaussian_wasserstein_commutative(
            policy, p, q, self.scale_prec
        )
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)
        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            eta = ch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.0
            eta = ch.max(-eta, eta)

            new_sqrt = (sqrt + ch.einsum("i,ijk->ijk", eta, old_sqrt)) / (
                1.0 + eta + 1e-16
            )[..., None, None]
            proj_sqrt = ch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        return proj_mean, proj_sqrt

    def trust_region_value(self, policy, p, q):
        return gaussian_wasserstein_commutative(
            policy, p, q, scale_prec=self.scale_prec
        )
