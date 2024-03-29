from mprl.env import EnvConfigGateway

from ..common.config_gateway import ModelConfigGateway
from ..physics.moe_prediction import MOEPrediction
from .moe import MixtureOfExperts


class MOEFactory:
    def __init__(
        self, config_gateway: ModelConfigGateway, env_config_gateway: EnvConfigGateway
    ):
        self._gateway = config_gateway
        self._env_config_gateway = env_config_gateway

    def create(self):
        config_model = self._gateway.get_model_config()
        model = MixtureOfExperts(
            state_dim_in=config_model.state_dim_in,
            state_dim_out=config_model.state_dim_out,
            action_dim=config_model.action_dim,
            num_experts=config_model.num_experts,
            network_width=config_model.network_width,
            variance=config_model.variance,
        )
        predictor = MOEPrediction(
            model,
            self._env_config_gateway.get_env_name(),
            naive_prepare=config_model.naive_state_input,
        )
        return predictor
