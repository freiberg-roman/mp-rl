from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MPRLEnvironment(ABC):
    @abstractmethod
    def step(self, action: np.array):
        raise NotImplementedError

    @abstractmethod
    def reset(self, time_out_after: int):
        raise NotImplementedError

    @abstractmethod
    def full_reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_sim_state(self) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    @abstractmethod
    def set_sim_state(self, sim_state: Tuple[np.array, np.array]):
        raise NotImplementedError

    @abstractmethod
    def render(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def close_viewer(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dt(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def total_steps(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def steps_after_reset(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def random_action(self) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def decompose_fn(
        self, states: np.ndarray, sim_states: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dof(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def get_info(self):
        return {}
