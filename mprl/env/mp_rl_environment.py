from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class MPRLEnvironment(ABC):
    @abstractmethod
    def step(self, action: np.array):
        raise NotImplementedError

    @abstractmethod
    def reset(self, timeout_after: int):
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

    @abstractmethod
    @property
    def dt(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def random_action(self) -> np.array:
        raise NotImplementedError
