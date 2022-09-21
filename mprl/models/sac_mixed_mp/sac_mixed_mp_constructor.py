from copy import deepcopy

import numpy as np
from mp_pytorch import MPFactory
from omegaconf import OmegaConf

from mprl.controllers.mp_trajectory import MPTrajectory
from mprl.env.config_gateway import EnvConfigGateway
from mprl.env.mj_factory import MujocoFactory
from mprl.utils import SequenceRB

from ...controllers import PDController
from ..common.config_gateway import ModelConfigGateway
from ..physics.ground_truth import GroundTruthPrediction
from .agent import SACMixedMP


class SACMixedMPFactory:
    def __init__(
        self, config_gateway: ModelConfigGateway, env_config_gateway: EnvConfigGateway
    ):
        self._gateway = config_gateway
        self._env_gateway = env_config_gateway

    def create(self):
        env_cfg = self._gateway.get_environment_config()
        buffer = SequenceRB(
            capacity=self._gateway.get_buffer_config().capacity,
            state_dim=env_cfg.state_dim,
            action_dim=env_cfg.action_dim,
            sim_qpos_dim=env_cfg.sim_qpos_dim,
            sim_qvel_dim=env_cfg.sim_qvel_dim,
        )
        cfg_net = self._gateway.get_network_config()
        cfg_hyper = self._gateway.get_hyper_parameter_config()
        cfg_model = self._gateway.get_model_config()
        env = MujocoFactory(self._env_gateway).create()
        if cfg_model.name == "off_policy":
            model = None
        elif cfg_model.name == "ground_truth":
            model = GroundTruthPrediction(
                env=MujocoFactory(env_config_gateway=self._env_gateway).create()
            )
        else:
            raise ValueError(f"Unknown model name {cfg_model.name}")

        idmp = MPFactory.init_mp(self._gateway.get_mp_config())
        planner = MPTrajectory(dt=env.dt, mp=idmp, device=self._gateway.get_device())
        pgains = np.array(self._gateway.get_ctrl_config().pgains)
        dgains = np.array(self._gateway.get_ctrl_config().dgains)
        pd_ctrl = PDController(
            pgains=pgains, dgains=dgains, device=self._gateway.get_device()
        )

        return SACMixedMP(
            buffer=buffer,
            state_dim=env_cfg.state_dim,
            action_dim=env_cfg.action_dim,
            network_width=cfg_net.network_width,
            network_depth=cfg_net.network_depth,
            lr=cfg_hyper.lr,
            gamma=cfg_hyper.gamma,
            tau=cfg_hyper.tau,
            alpha=cfg_hyper.alpha,
            batch_size=cfg_hyper.batch_size,
            device=self._gateway.get_device(),
            num_steps=cfg_hyper.num_steps,
            num_basis=cfg_hyper.num_basis,
            num_dof=cfg_hyper.num_dof,
            model=model,
            policy_loss_type=cfg_hyper.policy_loss,
            decompose_fn=env.decompose_fn,
            planner_act=deepcopy(planner),
            planner_update=deepcopy(planner),
            planner_eval=deepcopy(planner),
            ctrl=pd_ctrl,
        )
