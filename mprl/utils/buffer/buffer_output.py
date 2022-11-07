from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class EnvStep:
    """Used in SAC and SAC_MP"""

    state: np.ndarray
    next_state: np.ndarray
    action: np.ndarray
    reward: float
    done: bool

    def to_torch_batch(self):
        return (
            torch.from_numpy(self.state).to(torch.float32),
            torch.from_numpy(self.next_state).to(torch.float32),
            torch.from_numpy(self.action).to(torch.float32),
            torch.unsqueeze(torch.tensor(self.reward).to(torch.float32), dim=-1),
            torch.unsqueeze(torch.tensor(self.done), dim=-1),
        )


@dataclass
class EnvSequence:
    """Used in SAC_MIXED_MP and SAC_TR"""

    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    sim_states: Tuple[np.ndarray, np.ndarray]
    des_qpvs: Tuple[np.ndarray, np.ndarray]
    des_qpvs_next: Tuple[np.ndarray, np.ndarray]
    weight_means: np.ndarray
    weight_stds: np.ndarray
    idxes: np.ndarray

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
            (torch.from_numpy(self.des_qpvs[0]), torch.from_numpy(self.des_qpvs[1])),
            (
                torch.from_numpy(self.des_qpvs_next[0]),
                torch.from_numpy(self.des_qpvs_next[1]),
            ),
            torch.from_numpy(self.weight_means),
            torch.from_numpy(self.weight_stds),
            self.idxes,
        )
