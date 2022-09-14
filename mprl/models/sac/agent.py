from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import Parameter
from torch.optim import Adam

from mprl.utils.math_helper import hard_update

from ..common import Actable, Evaluable, Serializable, Trainable, QNetwork
from .networks import GaussianPolicy


class SAC(Actable, Evaluable, Serializable, Trainable):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        network_width: int,
        network_depth: int,
    ):

        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.device: torch.device = device


        # Networks
        self.critic: QNetwork = QNetwork((state_dim, action_dim), network_width, network_depth).to(
            device=self.device
        )
        self.critic_target: QNetwork = QNetwork((state_dim, action_dim), network_width, network_depth).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            state_dim, action_dim, hidden_size
        ).to(self.device)

    def select_action(
        self, state: np.ndarray, sim_state=None, evaluate: bool = False
    ) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0], {}

    def sample(self, state) -> torch.Tensor:
        return self.policy.sample(state)

    def parameters(self) -> Iterator[Parameter]:
        return self.policy.parameters()

    # Save model parameters
    def save(self, base_path: str, folder: str) -> None:
        path = base_path + folder + "/sac/"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            path + "model.pt",
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.policy.forward(states)

    # Load model parameters
    def load(self, path: str, evaluate: bool = False) -> None:
        ckpt_path = path + "/sac/model.pt"
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
