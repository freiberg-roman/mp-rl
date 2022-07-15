from abc import abstractmethod


class Prediction:
    @abstractmethod
    def next_state(self, states, actions):
        raise NotImplementedError
