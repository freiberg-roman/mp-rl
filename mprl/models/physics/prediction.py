from abc import abstractmethod


class Prediction:
    @abstractmethod
    def next_state(self, sim_state, actions):
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, batch: any):
        raise NotImplementedError
