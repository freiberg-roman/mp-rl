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
        network_cfg = self._config.alg.network
        cfg = OmegaConf.create(
            {
                **OmegaConf.to_container(self._config.alg),
                **OmegaConf.to_container(train_cfg),
                **OmegaConf.to_container(network_cfg),
            }
        )
        cfg.alpha = self._config.alg.alpha
        cfg.auto_alpha = self._config.alg.auto_alpha
        cfg.target_entropy = self._config.alg.target_entropy
        cfg.num_steps = self._config.alg.get("num_steps", 1)
        cfg.alpha_q = self._config.alg.get("alpha_q", 0.0)
        cfg.num_basis = self._config.alg.mp.get("num_basis", 1)
        cfg.num_dof = self._config.alg.mp.get("num_dof", 1)
        cfg.layer_type = self._config.alg.get("layer_type", "kl")
        cfg.mean_bound = self._config.alg.get(
            "mean_bound", 100.0
        )  # just to make it obvious
        cfg.cov_bound = self._config.alg.get("cov_bound", 100.0)
        cfg.use_imp_sampling = self._config.alg.get("use_imp_sampling", False)
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
        return self._config.alg.network

    def get_mp_config(self) -> DictConfig:
        """
        Returns the multiprocessing configuration.

        :return: The multiprocessing configuration.
        """
        return self._config.alg.mp

    def get_ctrl_config(self) -> DictConfig:
        """
        Returns the control configuration.

        :return: The control configuration.
        """
        return self._config.alg.ctrl

    def get_environment_config(self) -> DictConfig:
        """
        Returns the environment configuration.

        :return: The environment configuration.
        """
        return self._config.env

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
        train_cfg = self._config.train
        warm_start_steps = train_cfg.warm_start_steps
        cfg = OmegaConf.create(
            {
                **OmegaConf.to_container(train_cfg),
                **{"time_out_after": self._config.env.time_out_after},
            }
        )
        cfg.warm_start_steps = warm_start_steps
        return cfg

    def get_evaluation_config(self) -> DictConfig:
        """
        Returns the evaluation configuration.

        :return: The evaluation configuration.
        """
        return self._config.eval

    def get_model_config(self) -> DictConfig:
        """
        Returns the model configuration.

        :return: The model configuration.
        """
        return self._config.prediction
