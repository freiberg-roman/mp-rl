from mprl.env import EnvConfigGateway
from mprl.models import ModelConfigGateway
from mprl.models.physics.moe import MixtureOfExperts


class MOEFactory:
    def __init__(
        self, config_gateway: ModelConfigGateway, env_config_gateway: EnvConfigGateway
    ):
        self._gateway = config_gateway
        self._env_config_gateway = env_config_gateway

    @staticmethod
    def _get_prep_fn(name: str):
        if name == "HalfCheetah":
            return lambda x: x[..., 1:]
        elif "Meta" in name:
            ...  # TODO: Implement this
        else:
            raise ValueError(f"Unknown environment {name}")

    def create(self):
        config_model = self._gateway.get_model_config()
        env_name = self._env_config_gateway.get_env_name()
        fn = self._get_prep_fn(env_name)
        return MixtureOfExperts(
            state_dim_in=config_model.state_dim_in,
            state_dim_out=config_model.state_dim_out,
            action_dim=config_model.action_dim,
            num_experts=config_model.num_experts,
            network_width=config_model.network_width,
            variance=config_model.variance,
            prep_input_fn=fn,
            use_batch_normalization=config_model.use_batch_normalization,
        )
