from omegaconf.omegaconf import DictConfig


class ConfigRepository:
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

    def get_buffer_config(self) -> DictConfig:
        """
        Returns the buffer configuration.

        :return: The buffer configuration.
        """
        return self._config.buffer

    def get_algorithm_config(self) -> DictConfig:
        """
        Returns the agent configuration.

        :return: The agent configuration.
        """
        return self._config.alg

    def get_environment_config(self) -> DictConfig:
        """
        Returns the environment configuration.

        :return: The environment configuration.
        """
        return self._config.env

    def get_model_config(self) -> DictConfig:
        """
        Returns the model configuration.

        :return: The model configuration.
        """
        return self._config.model

    def get_logging_config(self) -> DictConfig:
        """
        Returns the logging configuration.

        :return: The logging configuration.
        """
        return self._config.logging
