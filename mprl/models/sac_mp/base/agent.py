from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.models.common.policy_network import GaussianPolicy
from mprl.utils import RandomRB, SequenceRB
from mprl.utils.math_helper import hard_update, soft_update

from ...common import QNetwork
from ..mp_agent import SACMPBase


class SACMP(SACMPBase):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        lr: float,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        num_steps: int,
        network_width: int,
        network_depth: int,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        ctrl: Controller,
        buffer: SequenceRB,
        decompose_fn: Callable,
    ):
        super().__init__(
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
        self.buffer: RandomRB = buffer
        self.batch_size: int = batch_size
        self.num_steps: int = num_steps
        weight_dim = (num_basis + 1) * action_dim

        # Networks
        del self.critic_target
        del self.critic
        del self.optimizer_critic

        self.critic: QNetwork = QNetwork(
            (state_dim, weight_dim), network_width, network_depth
        )
        self.critic_target: QNetwork = QNetwork(
            (state_dim, weight_dim), network_width, network_depth
        )
        hard_update(self.critic_target, self.critic)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)

        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, weight_dim),
            network_width,
            network_depth,
        )
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.c_state = None
        self.c_weights = None
        self.c_reward = 0.0
        self.c_done = False

    def sequence_reset(self):
        self.planner_act.reset_planner()
        self.c_state = None
        self.c_weights = None
        self.c_reward = 0.0
        self.c_done = False

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        self.c_reward += reward
        self.c_done = done

    def replan(self, state, sim_state, weights):
        if self.c_state is None:
            self.c_state = state
            self.c_weights = weights
            self.c_reward = 0.0
            self.c_done = False
            return

        self.buffer.add(
            self.c_state,
            state,
            self.c_weights,
            self.c_reward / self.num_steps,
            self.c_done,
            sim_state,
        )
        self.c_state = state
        self.c_weights = weights
        self.c_reward = 0.0
        self.c_done = False

    def sample(self, state):
        return self.policy.sample(state)

    def parameters(self):
        return self.policy.parameters()

    # Save model parameters
    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        ch.save(
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
            checkpoint = ch.load(path)
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
        states, next_states, actions, rewards, dones = self.buffer.sample_batch(
            self.batch_size
        )

        # Compute critic loss
        with ch.no_grad():
            next_state_action, next_state_log_pi = self.policy.sample_log_prob(
                next_states
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                ch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = rewards + (1 - dones.to(ch.float32)) * self.gamma * (
                min_qf_next_target
            )

        qf1, qf2 = self.critic(
            states, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,wt) - r(st,wt) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,wt) - r(st,wt) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        # Update critic
        self.optimizer_critic.zero_grad()
        qf_loss.backward()
        self.optimizer_critic.step()

        # Compute policy loss
        pi, log_pi = self.policy.sample_log_prob(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = ch.min(qf1_pi, qf2_pi)
        policy_loss = (
            (self.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # Update policy
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update target networks
        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": (-log_pi).detach().cpu().mean().item(),
        }
