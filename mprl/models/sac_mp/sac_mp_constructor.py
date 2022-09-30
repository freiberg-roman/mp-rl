from copy import deepcopy

import numpy as np
import torch
from mp_pytorch.mp import MPFactory

from mprl.controllers import MetaController, MPTrajectory, PDController
from mprl.env.config_gateway import EnvConfigGateway
from mprl.env.mj_factory import MujocoFactory
from mprl.utils import SequenceRB

from ..common.config_gateway import ModelConfigGateway
from .agent import SACMP


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
        env = MujocoFactory(self._env_gateway).create()

        cfg_idmp = self._gateway.get_mp_config()
        idmp = MPFactory.init_mp(
            mp_type="prodmp",
            mp_args=cfg_idmp.mp_args,
            num_dof=cfg_idmp.num_dof,
            tau=cfg_idmp.tau,
        )
        idmp.weights_scale = torch.tensor(cfg_idmp.mp_args.weight_scale)
        planner = MPTrajectory(dt=env.dt, mp=idmp, device=self._gateway.get_device())
        pgains = np.array(self._gateway.get_ctrl_config().pgains)

        is_pos_ctrl = "Pos" in self._env_gateway.get_env_name()
        if is_pos_ctrl:
            ctrl = MetaController(pgains=pgains)
        else:
            dgains = np.array(self._gateway.get_ctrl_config().dgains)
            ctrl = PDController(
                pgains=pgains, dgains=dgains, device=self._gateway.get_device()
            )

        return SACMP(
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
            decompose_fn=env.decompose_fn,
            planner_act=deepcopy(planner),
            planner_update=deepcopy(planner),
            planner_eval=deepcopy(planner),
            ctrl=ctrl,
        )
