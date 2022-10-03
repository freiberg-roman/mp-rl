from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mp_pytorch.util import tensor_linspace
from torch.optim import Adam

from mprl.utils import SequenceRB
from mprl.utils.ds_helper import to_np, to_ts
from mprl.utils.math_helper import hard_update, soft_update

from ...controllers import Controller, MPTrajectory
from .. import Actable, Evaluable, Serializable, Trainable
from ..common import QNetwork
from ..sac.networks import GaussianPolicy


class SACMP(Actable, Trainable, Serializable, Evaluable):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        num_steps: int,
        lr: float,
        batch_size: int,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        num_dof: int,
        network_width: int,
        network_depth: int,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        planner_update: MPTrajectory,
        ctrl: Controller,
        buffer: SequenceRB,
        decompose_fn: Callable,
    ):
        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.num_steps: int = num_steps
        self.device: torch.device = device
        self.buffer: SequenceRB = buffer
        self.planner_act: MPTrajectory = planner_act
        self.planner_eval: MPTrajectory = planner_eval
        self.planner_update: MPTrajectory = planner_update
        self.ctrl: Controller = ctrl
        self.decompose_fn: Callable = decompose_fn
        self.batch_size: int = batch_size
        self._current_weights: torch.Tensor = None
        self._current_time: int = -1
        self.num_dof: int = num_dof
        action_dim = (num_basis + 1) * num_dof + 1

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(device=self.device)
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, (num_basis + 1) * num_dof),
            network_width,
            network_depth,
            action_scale=1000.0,
        ).to(self.device)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)

    def select_weights_and_time(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            weight_times, _, _, _ = self.policy.sample(state)
        else:
            _, _, weight_times, _ = self.policy.sample(state)
        return weight_times, {
            "time": weight_times.squeeze()[-1].detach().cpu().numpy(),
        }

    def sequence_reset(self):
        if len(self.buffer) > 0:
            self.buffer.close_trajectory()
        self.planner_act.reset_planner()

    @torch.no_grad()
    def action(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = torch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = torch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_act)
        except StopIteration:
            weights, _, _ = self.policy.sample(state)
            self._current_weights = weights
            self._current_time = self.num_steps
            self.planner_act.init(weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps)
            q, v = next(self.planner_act)
        action = self.ctrl.get_action(q, v, b_q, b_v)
        self._current_time -= 1
        return to_np(action.squeeze())

    def eval_reset(self) -> np.ndarray:
        self.planner_eval.reset_planner()

    def eval_log(self) -> Dict:
        times = tensor_linspace(
            0,
            self.planner_eval.dt * self.planner_eval.num_t,
            self.planner_eval.num_t + 1,
        )
        pos = self.planner_eval.current_traj
        if pos is None:
            return {}
        plt.close()
        for dim in range(self.num_dof):
            plt.plot(times, pos[:, dim])
        return {"traj": plt}

    @torch.no_grad()
    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = torch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = torch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_eval)
        except StopIteration:
            _, _, weights = self.policy.sample(state)
            self.planner_eval.init(
                weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
            )
            q, v = next(self.planner_eval)
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return to_np(action.squeeze())

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        time = np.array([self._current_time]) + np.random.uniform(
            low=0.0, high=1.0
        )  # smear time
        weight_time = np.concatenate(
            (self._current_weights.squeeze().cpu().numpy(), time)
        )
        self.buffer.add(state, next_state, weight_time, reward, done, sim_state)

    def sample(self, state):
        return self.policy.sample(state)

    def parameters(self):
        return self.policy.parameters()

    # Save model parameters
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.optimizer_critic.state_dict(),
                "policy_optimizer_state_dict": self.optimizer_policy.state_dict(),
            },
            path + "model.pt",
        )

    # Load model parameters
    def load(
        self,
        path,
    ):
        if path is not None:
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.optimizer_critic.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            self.optimizer_policy.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )

    def set_eval(self):
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()

    def set_train(self):
        self.policy.train()
        self.critic.train()
        self.critic_target.train()

    def update(self) -> dict:
        batch = next(
            self.buffer.get_true_k_sequence_iter(1, self.num_steps, self.batch_size)
        ).to_torch_batch()
        # dimensions (batch_size, sequence_len, data_dimension)
        states, next_states, actions, rewards, dones, sim_states = batch

        next_states_idx = torch.floor(actions[:, 0, -1]).to(
            torch.long
        )  # get the time left for the action
        # TODO: think of version without for loop
        next_states_reduced = torch.zeros_like(next_states[:, 0, :])
        rewards_reduced = torch.zeros_like(rewards[:, 0, :])
        dones_reduced = torch.zeros_like(dones[:, 0, :])
        for i, idx in zip(range(self.batch_size), next_states_idx):
            next_states_reduced[i, :] = next_states[i, idx, :]
            rewards_reduced[i, :] = rewards[i, idx, :]
            dones_reduced[i, :] = dones[i, idx, :]

        next_states = next_states_reduced
        states = states[:, 0, :]
        actions = actions[:, 0, :]
        rewards = rewards_reduced
        dones = dones_reduced
        # Compute critic loss
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.sample(next_states)
            time = (
                torch.rand((self.batch_size, 1)) * self.num_steps
            )  # agent has no control over time
            next_state_action = torch.cat((next_state_action, time), dim=1)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = rewards + (1 - dones.to(torch.float32)) * self.gamma * (
                min_qf_next_target
            )

        qf1, qf2 = self.critic(
            states, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # Update critic
        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        self.optimizer_critic.step()

        # Compute policy loss
        weights, log_pi, _ = self.sample(states)
        time = (
            torch.rand((self.batch_size, 1)) * self.num_steps
        )  # agent has no control over time
        weights_time = torch.cat((weights, time), dim=1)
        qf1_pi, qf2_pi = self.critic(states, weights_time)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        # Update policy
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        soft_update(self.critic_target, self.critic, self.tau)
        return {
            "qf_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
            # weight statistics
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_max": weights[..., :-1].detach().cpu().max().item(),
            "weight_min": weights[..., :-1].detach().cpu().min().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weight_goal_max": weights[..., -1].detach().cpu().max().item(),
            "weight_goal_min": weights[..., -1].detach().cpu().min().item(),
        }
