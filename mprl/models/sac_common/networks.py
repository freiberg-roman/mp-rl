from typing import Tuple

import torch
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
        num_inputs: int,
        num_actions: int,
        hidden_dim: int,
        additional_actions: int = 0,
    ):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1: nn.Linear = nn.Linear(
            num_inputs + num_actions + additional_actions, hidden_dim
        )
        self.linear2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear3: nn.Linear = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4: nn.Linear = nn.Linear(
            num_inputs + num_actions + additional_actions, hidden_dim
        )
        self.linear5: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear6: nn.Linear = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], 1)

        x1 = F.silu(self.linear1(xu))
        x1 = F.silu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.silu(self.linear4(xu))
        x2 = F.silu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
