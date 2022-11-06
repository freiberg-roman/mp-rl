from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch as ch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam

from mprl.models.common.policy_network import GaussianPolicy
from mprl.utils import RandomRB
from mprl.utils.math_helper import hard_update, soft_update

from ...utils.serializable import Serializable
from ..common import Actable, Evaluable, QNetwork, Trainable


class SAC(Actable, Evaluable, Serializable, Trainable):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        automatic_entropy_tuning: bool,
        lr: float,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        network_width: int,
        network_depth: int,
        buffer: RandomRB,
    ):

        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.buffer = buffer
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        )
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        )
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, action_dim), network_width, network_depth
        )
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = ch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def sequence_reset(self):
        """Reset the internal state of the agent. In this case, the agent is stateless."""
        pass

    @ch.no_grad()
    def action_train(self, state: np.ndarray, info) -> np.ndarray:
        _ = info  # not used
        state = ch.FloatTensor(state).unsqueeze(0)
        action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def eval_reset(self) -> np.ndarray:
        """Reset the internal state of the agent. In this case, the agent is stateless."""
        pass

    def eval_log(self) -> Dict:
        return {}

    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        _ = info
        state = ch.FloatTensor(state).unsqueeze(0)
        mean_action = self.policy.mean(state)
        return mean_action.detach().cpu().numpy()[0]

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        self.buffer.add(state, next_state, action, reward, done, sim_state)

    def parameters(self) -> Iterator[Parameter]:
        return self.policy.parameters()

    def store_under(self):
        return "sac"

    # Save model parameters
    def store(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        ch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "optimizer_critic_state_dict": self.optimizer_critic.state_dict(),
                "optimizer_policy_state_dict": self.optimizer_policy.state_dict(),
                **(
                    {}
                    if self.automatic_entropy_tuning
                    else {"optimizer_entropy": self.alpha_optim.state_dict()}
                ),
            },
            path + "/model.pt",
        )
        self.buffer.store(path + "/" + self.buffer.store_under())

    # Load model parameters
    def load(self, path: str) -> None:
        ckpt_path = path + "/model.pt"
        if ckpt_path is not None:
            checkpoint = ch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.optimizer_critic.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            self.optimizer_policy.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )
            if self.automatic_entropy_tuning:
                self.alpha_optim.load_state_dict(checkpoint["optimizer_entropy"])
        self.buffer.load(path + "/" + self.buffer.store_under())

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
            batch_size=self.batch_size
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

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # Update target networks
        soft_update(self.critic_target, self.critic, self.tau)

        return {
            "critic_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }
