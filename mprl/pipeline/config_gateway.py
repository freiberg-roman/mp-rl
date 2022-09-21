from abc import ABC, abstractmethod

from omegaconf.omegaconf import DictConfig


class TrainConfigGateway(ABC):
    @abstractmethod
    def get_training_config(self) -> DictConfig:
        raise NotImplementedError

    @abstractmethod
    def get_evaluation_config(self) -> DictConfig:
        raise NotImplementedError
