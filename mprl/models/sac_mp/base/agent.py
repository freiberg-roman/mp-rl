from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.models.common.interfaces import Actable, Evaluable, Serializable, Trainable
from mprl.models.common.policy_network import GaussianPolicy
from mprl.utils import SequenceRB
from mprl.utils.math_helper import hard_update, soft_update

from ..mp_agent import SACMPBase


class SACMP(SACMPBase):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        lr: float,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        network_width: int,
        network_depth: int,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        planner_update: MPTrajectory,
        ctrl: Controller,
        buffer: SequenceRB,
        decompose_fn: Callable,
    ):
        super(SACMP).__init__(
            action_dim,
            ctrl,
            decompose_fn,
            lr,
            network_width,
            network_depth,
            planner_act,
            planner_eval,
            state_dim,
        )
        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.buffer: SequenceRB = buffer
        self.planner_update: MPTrajectory = planner_update
        weight_dim = (num_basis + 1) * action_dim + 1

        # Networks
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, weight_dim),
            network_width,
            network_depth,
        ).to(self.device)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)

    def sequence_reset(self):
        if len(self.buffer) > 0:
            self.buffer.close_trajectory()
        self.planner_act.reset_planner()

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        self.buffer.add(state, next_state, self.c_weights, reward, done, sim_state)

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
        # dimensions (batch_size, sequence_len, data_dimension)
        batch = self.buffer.get_random_batch().to_torch_batch()
        states, next_states, actions, rewards, dones = batch
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
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
        }
