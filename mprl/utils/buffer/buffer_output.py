from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class EnvStep:
    state: np.ndarray
    next_state: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    sim_state: Tuple[np.ndarray, np.ndarray]

    def to_torch_batch(self):
        return (
            torch.unsqueeze(torch.from_numpy(self.state).to(torch.float32), 0),
            torch.unsqueeze(torch.from_numpy(self.next_state).to(torch.float32), 0),
            torch.unsqueeze(torch.from_numpy(self.action).to(torch.float32), 0),
            torch.unsqueeze(torch.tensor(self.reward).to(torch.float32), 0),
            torch.unsqueeze(torch.tensor(self.done), 0),
            self.sim_state,  # they won't be used in torch
        )


@dataclass
class EnvSteps:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    sim_states: Tuple[np.ndarray, np.ndarray]

    def __len__(self):
        return len(self.states)

    def to_torch_batch(self):
        return (
            torch.from_numpy(self.states),
            torch.from_numpy(self.next_states),
            torch.from_numpy(self.actions),
            torch.unsqueeze(torch.from_numpy(self.rewards), dim=-1),
            torch.unsqueeze(torch.from_numpy(self.dones), dim=-1),
            self.sim_states,  # they won't be used in torch
        )
