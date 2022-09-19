from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig

from mprl.env import EnvConfigGateway
from mprl.models import ModelConfigGateway
from mprl.pipeline.config_gateway import TrainConfigGateway


class ConfigRepository(ModelConfigGateway, EnvConfigGateway, TrainConfigGateway):
    """
    Manages the configuration of the project.
    This allows hydras configuration to be structured from user perspective, but still be accessible
    from a single point in the code and restructuring it as needed.
    """

    def __init__(self, project_configuration: DictConfig):
        """
        Initializes the configuration repository. This is the only place where the configuration is
        stored. All other classes should access the configuration through this class.

        :param project_configuration:
        """
        self._config = project_configuration

    def get_config(self) -> DictConfig:
        """
        Returns the configuration.

        :return: The configuration.
        """
        return self._config

    def get_config_for(self, key: str) -> DictConfig:
        """
        Returns the configuration for the given key.

        :param key: The key to get the configuration for.
        :return: The configuration for the given key.
        """
        return self._config[key]

    def get_hyper_parameter_config(self) -> DictConfig:
        """
        Returns the hyperparameters configuration.

        :return: The hyperparameters configuration.
        """
        train_cfg = self._config.train
        cfg = OmegaConf.create(
            {
                **OmegaConf.to_container(self._config.alg.agent.hyper_parameter),
                **OmegaConf.to_container(train_cfg),
            }
        )
        cfg.alpha = self._config.alg.agent.hyper_parameter.alpha
        return cfg

    def get_buffer_config(self) -> DictConfig:
        """
        Returns the buffer configuration.

        :return: The buffer configuration.
        """
        return self._config.buffer

    def get_network_config(self) -> DictConfig:
        """
        Returns the network configuration.

        :return: The network configuration.
        """
        return self._config.alg.agent.network

    def get_environment_config(self) -> DictConfig:
        """
        Returns the environment configuration.

        :return: The environment configuration.
        """
        return self._config.env

    def get_device(self) -> str:
        """
        Returns the device to use.

        :return: The device to use.
        """
        return self._config.device

    def get_env_name(self) -> str:
        """
        Returns the name of the environment.

        :return: The name of the environment.
        """
        return self._config.env.name

    def get_training_config(self) -> DictConfig:
        """
        Returns the training configuration.

        :return: The training configuration.
        """
        return self._config.train

    def get_evaluation_config(self) -> DictConfig:
        """
        Returns the evaluation configuration.

        :return: The evaluation configuration.
        """
        return self._config.eval
