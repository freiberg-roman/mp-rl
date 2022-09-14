from mprl.config import ConfigRepository


class SACFactory:
    def __init__(self, config_repository: ConfigRepository):
        self.config_repository = config_repository

    def create(self):
        pass
