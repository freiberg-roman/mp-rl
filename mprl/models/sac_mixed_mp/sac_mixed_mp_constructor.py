from copy import deepcopy

import numpy as np
import torch
from mp_pytorch.basis_gn import ProDMPBasisGenerator
from mp_pytorch.mp import ProDMP
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator

from mprl.controllers import MetaController, MPTrajectory, PDController
from mprl.env.config_gateway import EnvConfigGateway
from mprl.env.mj_factory import MujocoFactory
from mprl.utils import SequenceRB

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

        cfg_idmp = self._gateway.get_mp_config()

        # Build ProDMP Controller
        phase_gn = ExpDecayPhaseGenerator(
            tau=cfg_idmp.tau,
            delay=0.0,
            learn_tau=False,
            learn_delay=False,
            alpha_phase=cfg_idmp.mp_args["alpha_phase"],
            dtype=torch.float32,
            device=self._gateway.get_device(),
        )
        basis_gn = ProDMPBasisGenerator(
            phase_generator=phase_gn,
            num_basis=cfg_idmp.mp_args["num_basis"],
            basis_bandwidth_factor=cfg_idmp.mp_args["basis_bandwidth_factor"],
            num_basis_outside=cfg_idmp.mp_args["num_basis_outside"],
            dt=cfg_idmp.mp_args["dt"],
            alpha=cfg_idmp.mp_args["alpha"],
            dtype=torch.float32,
            device=self._gateway.get_device(),
        )
        idmp = ProDMP(
            basis_gn=basis_gn,
            num_dof=cfg_idmp.num_dof,
            dtype=torch.float32,
            device=self._gateway.get_device(),
            weights_scale=cfg_idmp.mp_args["weight_scale"],
            **cfg_idmp.mp_args,
        )
        planner = MPTrajectory(dt=env.dt, mp=idmp, device=self._gateway.get_device())
        # Build evaluation controller
        phase_gn_eval = ExpDecayPhaseGenerator(
            tau=cfg_idmp.tau_eval,
            delay=0.0,
            learn_tau=False,
            learn_delay=False,
            alpha_phase=cfg_idmp.mp_args["alpha_phase"],
            dtype=torch.float32,
            device=self._gateway.get_device(),
        )
        basis_gn_eval = ProDMPBasisGenerator(
            phase_generator=phase_gn_eval,
            num_basis=cfg_idmp.mp_args["num_basis"],
            basis_bandwidth_factor=cfg_idmp.mp_args["basis_bandwidth_factor"],
            num_basis_outside=cfg_idmp.mp_args["num_basis_outside"],
            dt=cfg_idmp.mp_args["dt"],
            alpha=cfg_idmp.mp_args["alpha"],
            dtype=torch.float32,
            device=self._gateway.get_device(),
        )
        idmp_eval = ProDMP(
            basis_gn=basis_gn_eval,
            num_dof=cfg_idmp.num_dof,
            dtype=torch.float32,
            device=self._gateway.get_device(),
            weights_scale=cfg_idmp.mp_args["weight_scale"],
            **cfg_idmp.mp_args,
        )
        pgains = np.array(self._gateway.get_ctrl_config().pgains)
        _ = MPTrajectory(
            dt=env.dt, mp=idmp_eval, device=self._gateway.get_device()
        )

        is_pos_ctrl = "Pos" in self._env_gateway.get_env_name()
        if is_pos_ctrl:
            ctrl = MetaController(pgains=pgains)
        else:
            dgains = np.array(self._gateway.get_ctrl_config().dgains)
            ctrl = PDController(
                pgains=pgains, dgains=dgains, device=self._gateway.get_device()
            )

        return SACMixedMP(
            buffer=buffer,
            state_dim=env_cfg.state_dim,
            action_dim=env_cfg.action_dim,
            network_width=cfg_net.network_width,
            network_depth=cfg_net.network_depth,
            action_scale=cfg_net.action_scale,
            lr=cfg_hyper.lr,
            gamma=cfg_hyper.gamma,
            tau=cfg_hyper.tau,
            alpha=cfg_hyper.alpha,
            alpha_q=cfg_hyper.alpha_q,
            batch_size=cfg_hyper.batch_size,
            device=self._gateway.get_device(),
            num_steps=cfg_hyper.num_steps,
            num_steps_eval=cfg_hyper.num_steps_eval,
            num_basis=cfg_hyper.num_basis,
            num_dof=cfg_hyper.num_dof,
            model=model,
            policy_loss_type=cfg_hyper.policy_loss,
            decompose_fn=env.decompose_fn,
            planner_act=deepcopy(planner),
            planner_update=deepcopy(planner),
            planner_eval=deepcopy(planner),
            ctrl=ctrl,
        )
