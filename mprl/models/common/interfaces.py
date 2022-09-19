from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Actable(ABC):
    @abstractmethod
    def sequence_reset(self):
        raise NotImplementedError

    @abstractmethod
    def action(self, state: np.ndarray, info: any) -> np.ndarray:
        raise NotImplementedError


class Evaluable(ABC):
    @abstractmethod
    def eval_reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        raise NotImplementedError


class Trainable(ABC):
    @abstractmethod
    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError


class Predictable(ABC):
    @abstractmethod
    def next_state(self, sim_state, actions):
        raise NotImplementedError


class Serializable(ABC):
    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError

    @abstractmethod
    def set_eval(self):
        raise NotImplementedError

    @abstractmethod
    def set_train(self):
        raise NotImplementedError
