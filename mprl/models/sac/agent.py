from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam

from mprl.utils import RandomRB
from mprl.utils.math_helper import hard_update, soft_update

from ..common import Actable, Evaluable, QNetwork, Serializable, Trainable
from .networks import GaussianPolicy


class SAC(Actable, Evaluable, Serializable, Trainable):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        automatic_entropy_tuning: bool,
        lr: float,
        batch_size: int,
        device: torch.device,
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
        self.device: torch.device = device
        self.buffer = buffer
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(device=self.device)
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def sequence_reset(self):
        """Reset the internal state of the agent. In this case, the agent is stateless."""
        pass

    @torch.no_grad()
    def action(self, state: np.ndarray, info) -> np.ndarray:
        _ = info  # not used
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def eval_reset(self) -> np.ndarray:
        """Reset the internal state of the agent. In this case, the agent is stateless."""
        pass

    def eval_log(self) -> Dict:
        return {}

    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        _ = info
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

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

    # Save model parameters
    def save(self, path: str) -> None:
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
    def load(self, path: str) -> None:
        ckpt_path = path + "/sac/model.pt"
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
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
        batch = next(self.buffer.get_iter(1, self.batch_size)).to_torch_batch()
        states, next_states, actions, rewards, dones, _ = batch

        # Compute critic loss
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
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
        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
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
