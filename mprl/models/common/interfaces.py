from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class Actable(ABC):
    @abstractmethod
    def sequence_reset(self):
        raise NotImplementedError

    @abstractmethod
    def action_train(self, state: np.ndarray, info: any) -> np.ndarray:
        raise NotImplementedError


class Evaluable(ABC):
    @abstractmethod
    def eval_reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def eval_log(self) -> Dict:
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
    def next_state(self, states, actions, sim_states=None):
        raise NotImplementedError
