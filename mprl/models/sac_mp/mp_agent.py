from abc import ABC
from typing import Callable, Dict

import numpy as np
import torch as ch
import wandb
from matplotlib import pyplot as plt
from mp_pytorch.util import tensor_linspace
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.models import Actable, Evaluable, Serializable, Trainable
from mprl.models.common import QNetwork
from mprl.utils.ds_helper import to_np
from mprl.utils.math_helper import hard_update


class SACMPBase(ABC, Actable, Evaluable, Serializable, Trainable):
    def __init__(
        self,
        action_dim: int,
        ctrl: Controller,
        decompose_fn: Callable,
        lr: float,
        network_width: int,
        network_depth: int,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        state_dim: int,
    ):
        # Parameters
        self.planner_act: MPTrajectory = planner_act
        self.planner_eval: MPTrajectory = planner_eval
        self.ctrl: Controller = ctrl
        self.decompose_fn: Callable = decompose_fn

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(device=self.device)
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)

        self.c_des_q = None
        self.c_des_v = None

        # Logging variables
        self.weights_log = []
        self.traj_des_log = []
        self.traj_log = []

    def sample(self, state: ch.Tensor) -> ch.Tensor:
        """Should just return the sampled policy actions, not the log probability."""
        raise NotImplementedError

    def replan(self, state, sim_state, weights):
        """Called at the beginning of each replan."""
        pass

    @ch.no_grad()
    def action_train(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = ch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = ch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = ch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_act)
            self.c_des_q = q
            self.c_des_v = v
            action = self.ctrl.get_action(q, v, b_q, b_v)
        except StopIteration:
            weights = self.sample(state)
            b_q_des, b_v_des = self.planner_act.get_next_bc()
            if b_q_des is not None and b_v_des is not None:
                self.planner_act.init(
                    weights,
                    bc_pos=b_q_des[None],
                    bc_vel=b_v_des[None],
                    num_t=self.num_steps,
                )
            else:
                self.planner_act.init(
                    weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
                )
            q, v = next(self.planner_act)
            self.c_des_q = q
            self.c_des_v = v
            action = self.ctrl.get_action(q, v, b_q, b_v)
            self.replan(state, sim_state, weights)

        return to_np(action.squeeze())

    def eval_reset(self) -> np.ndarray:
        self.planner_eval.reset_planner()
        self.weights_log = []
        self.traj_log = []
        self.traj_des_log = []

    def eval_log(self) -> Dict:
        times_full_traj = tensor_linspace(
            0.0, self.planner_eval.dt * len(self.traj_log), len(self.traj_log)
        )
        # just last planned trajectory
        times = tensor_linspace(
            0,
            self.planner_eval.dt * self.planner_eval.num_t,
            self.planner_eval.num_t + 1,
        )
        pos = self.planner_eval.current_traj
        if pos is None:
            return {}
        plt.close("all")
        figs = {}

        for dim in range(self.num_dof):
            figs["last_desired_traj"] = plt.figure()
            ax = figs["last_desired_traj"].add_subplot(111)
            ax.plot(times, pos[:, dim])

        pos_real = np.array(self.traj_log)
        pos_des = np.array(self.traj_des_log)
        for dim in range(self.num_dof):
            figs["traj_" + str(dim)] = plt.figure()
            ax = figs["traj_" + str(dim)].add_subplot(111)
            ax.plot(times_full_traj, pos_real[:, dim])
            ax.plot(times_full_traj, pos_des[:, dim])

        pos_delta = np.abs(pos_real - pos_des)
        plt.figure()
        for dim in range(self.num_dof):
            figs["abs_delta_traj_" + str(dim)] = plt.figure()
            ax = figs["abs_delta_traj_" + str(dim)].add_subplot(111)
            ax.plot(times_full_traj, pos_delta[:, dim])
        return {
            **figs,
            "weights_histogram": wandb.Histogram(np.array(self.weights_log).flatten()),
        }

    @ch.no_grad()
    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = ch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = ch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = ch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_eval)
        except StopIteration:
            weights = self.sample(state)
            b_q_des, b_v_des = self.planner_eval.get_next_bc()
            if b_q_des is not None and b_v_des is not None:
                self.planner_eval.init(
                    weights,
                    bc_pos=b_q_des[None],
                    bc_vel=b_v_des[None],
                    num_t=self.num_steps,
                )
            else:
                self.planner_eval.init(
                    weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
                )
            q, v = next(self.planner_eval)
            self.weights_log.append(to_np(weights.squeeze()).flatten())
        q_curr = self.planner_eval.get_current()
        self.traj_des_log.append(to_np((q_curr)))
        self.traj_log.append(to_np(b_q[0]))
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return to_np(action.squeeze())
