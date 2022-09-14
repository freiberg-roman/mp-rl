from abc import ABC, abstractmethod

import numpy as np


class Actable(ABC):
    @abstractmethod
    def action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Evaluable(ABC):
    @abstractmethod
    def eval_reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def action_eval(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Trainable(ABC):
    @abstractmethod
    def update(self):
        raise NotImplementedError


class Prediction(ABC):
    @abstractmethod
    def next_state(self, sim_state, actions):
        raise NotImplementedError


class Serializable(ABC):
    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError
