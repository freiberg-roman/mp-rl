from mprl.config import ConfigRepository


class SACConfigGateway:
    """Access to the SAC configuration."""

    def __init__(self, config_repository: ConfigRepository):
        # note: should use dependency injection in future
        self._config_repository = config_repository


class SACFactory:
    def __init__(self, config_gateway: SACConfigGateway):
        self._config_gateway = config_gateway

    def create(self):
        pass
