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

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import torch as ch
import torch.nn as nn

from mprl.trd_party.trl.trust_region_projections.utils.lib import inverse_softplus


class AbstractGaussianPolicy(nn.Module, ABC):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        init: str = "orthogonal",
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        contextual_std: bool = False,
        init_std: float = 1.0,
        minimal_std: float = 1e-5,
        share_weights: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.contextual_std = contextual_std
        self.share_weights = share_weights
        self.minimal_std = minimal_std
        self.init_std = ch.tensor(init_std)

        prev_size = hidden_sizes[-1]

        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus

        # This shift is applied to the Parameter/cov NN output before applying the transformation
        # and gives hence the wanted initial cov
        self._pre_activation_shift = self._get_preactivation_shift(
            self.init_std, minimal_std
        )
        self._mean = self._get_mean(action_dim, prev_size, init)
        self._pre_std = self._get_std(contextual_std, action_dim, prev_size, init)

    @abstractmethod
    def forward(self, x, train=True):
        pass

    def get_value(self, x, train=True):
        if self.share_weights:
            self.train(train)
            for affine in self.affine_layers:
                x = self.activation(affine(x))
            value = self.final_value(x)
        elif self.vf_model:
            value = self.vf_model(x, train)
        else:
            raise ValueError(
                "Must be sharing weights or use joint training to use get_value."
            )

        return value

    def squash(self, x):
        return x

    def _get_mean(self, action_dim, prev_size=None, init=None, scale=0.01):
        mean = nn.Linear(prev_size, action_dim)
        return mean

    # @final
    def _get_std(self, contextual_std: bool, action_dim, prev_size=None, init=None):
        if contextual_std:
            return self._get_std_layer(prev_size, action_dim, init)
        else:
            return self._get_std_parameter(action_dim)

    def _get_preactivation_shift(self, init_std, minimal_std):
        return self.diag_activation_inv(init_std - minimal_std)

    @abstractmethod
    def _get_std_parameter(self, action_dim):
        pass

    @abstractmethod
    def _get_std_layer(self, prev_size, action_dim, init):
        pass

    @abstractmethod
    def sample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        pass

    @abstractmethod
    def rsample(self, p: Tuple[ch.Tensor, ch.Tensor], n=1) -> ch.Tensor:
        pass

    @abstractmethod
    def log_probability(
        self, p: Tuple[ch.Tensor, ch.Tensor], x: ch.Tensor, **kwargs
    ) -> ch.Tensor:
        pass

    @abstractmethod
    def entropy(self, p: Tuple[ch.Tensor, ch.Tensor]) -> ch.Tensor:
        pass

    @abstractmethod
    def log_determinant(self, std: ch.Tensor) -> ch.Tensor:
        pass

    @abstractmethod
    def maha(self, mean, mean_other, std) -> ch.Tensor:
        pass

    @abstractmethod
    def precision(self, std: ch.Tensor) -> ch.Tensor:
        pass

    @abstractmethod
    def covariance(self, std) -> ch.Tensor:
        pass

    @abstractmethod
    def set_std(self, std: ch.Tensor) -> None:
        pass

    def get_last_layer(self):
        return self._affine_layers[-1].weight.data

    def papi_weight_update(self, eta: ch.Tensor, A: ch.Tensor):
        self._affine_layers[-1].weight.data *= eta
        self._affine_layers[-1].weight.data += (1 - eta) * A

    @property
    def is_root(self):
        return False

    @property
    def is_diag(self):
        return False
