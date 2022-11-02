from abc import ABC, abstractmethod

from omegaconf.omegaconf import DictConfig


class ModelConfigGateway(ABC):
    @abstractmethod
    def get_hyper_parameter_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_network_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_buffer_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_environment_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_model_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_mp_config(self) -> DictConfig:
        pass

    @abstractmethod
    def get_ctrl_config(self) -> DictConfig:
        pass
