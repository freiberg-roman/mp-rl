from abc import abstractmethod

import torch


class Prediction:
    @abstractmethod
    def next_state(self, states, actions):
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, batch: any):
        raise NotImplementedError
