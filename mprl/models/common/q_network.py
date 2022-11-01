from typing import Tuple

import torch as ch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m: nn.Linear):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        network_width: int,
        network_depth: int,
    ):
        super(QNetwork, self).__init__()
        state_dim, action_dim = input_dim
        self.first_input_layer = nn.Linear(state_dim + action_dim, network_width)
        self.second_input_layer = nn.Linear(state_dim + action_dim, network_width)

        self.pipeline_one = []
        self.pipeline_two = []
        # Q1, Q2 architectures
        for _ in range(network_depth - 1):
            self.pipeline_one.append(nn.Linear(network_width, network_width))
            self.pipeline_two.append(nn.Linear(network_width, network_width))

        self.pipeline_one = nn.ModuleList(self.pipeline_one)
        self.pipeline_two = nn.ModuleList(self.pipeline_two)
        self.first_output_layer = nn.Linear(network_width, 1)
        self.second_output_layer = nn.Linear(network_width, 1)

        self.apply(weights_init_)

    def forward(
        self, state: ch.Tensor, action: ch.Tensor
    ) -> Tuple[ch.Tensor, ch.Tensor]:
        xu = ch.cat([state, action], 1)

        x1, x2 = F.silu(self.first_input_layer(xu)), F.silu(self.second_input_layer(xu))
        for l1, l2 in zip(self.pipeline_one, self.pipeline_two):
            x1 = F.silu(l1(x1))
            x2 = F.silu(l2(x2))

        return self.first_output_layer(x1), self.second_output_layer(x2)
