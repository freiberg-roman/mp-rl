from dependency_injector import containers
from dependency_injector.wiring import Provider, inject

from mprl.utils import RandomRB

from ..common.config_gateway import ModelConfigGateway
from .agent import SAC


class SACFactory:
    @inject
    def __init__(
        self, gateway: ModelConfigGateway = Provider[containers.model_config_gateway]
    ):
        self.gateway = gateway

    def create(self):
        env_cfg = self.gateway.get_env_parameter_config()
        buffer = RandomRB(
            capacity=self.config_repository.get_buffer_config.capacity,
            state_dim=env_cfg.state_dim,
            action_dim=env_cfg.action_dim,
            sim_qpos_dim=env_cfg.sim_qpos_dim,
            sim_qvel_dim=env_cfg.sim_qvel_dim,
        )
        cfg_net = self.config_repository.get_network_config
        cfg_hyper = self.config_repository.get_hyperparameters_config
        return SAC(
            buffer=buffer,
            state_dim=env_cfg.state_dim,
            action_dim=env_cfg.action_dim,
            network_width=cfg_net.network_width,
            network_depth=cfg_net.depth,
            lr=cfg_hyper.lr,
            gamma=cfg_hyper.gamma,
            tau=cfg_hyper.tau,
            alpha=cfg_hyper.alpha,
            device=self.config_repository.get_device,
        )
