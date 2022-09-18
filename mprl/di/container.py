from dependency_injector import containers, providers

from mprl.config import ConfigRepository


class Container(containers.DeclarativeContainer):
    """Container for dependency injection"""

    config = providers.Configuration()
    model_config_gateway = providers.Factory(
        ConfigRepository,
        project_configuration=config.hydra_configuration,
    )
    env_config_gateway = providers.Factory(
        ConfigRepository,
        project_configuration=config.hydra_configuration,
    )
    train_scheme_gateway = providers.Factory(
        ConfigRepository, project_configuration=config.hydra_configuration
    )
