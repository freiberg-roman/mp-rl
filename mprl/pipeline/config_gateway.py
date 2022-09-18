from omegaconf.omegaconf import DictConfig


class TrainConfigGateway:
    def get_train_scheme(self) -> DictConfig:
        raise NotImplementedError
