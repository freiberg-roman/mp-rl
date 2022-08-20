from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import Parameter
from torch.optim import Adam

from mprl.models.sac_common.networks import QNetwork
from mprl.utils.math_helper import hard_update

from .networks import GaussianPolicy


class SAC:
    def __init__(self, cfg: OmegaConf):
        # Parameters
        self.gamma: float = cfg.gamma
        self.tau: float = cfg.tau
        self.alpha: float = cfg.alpha
        self.automatic_entropy_tuning: bool = cfg.automatic_entropy_tuning
        self.device: torch.device = torch.device(
            "cuda" if cfg.device == "cuda" else "cpu"
        )
        state_dim: int = cfg.env.state_dim
        action_dim: int = cfg.env.action_dim
        hidden_size: int = cfg.hidden_size

        # Networks
        self.critic: QNetwork = QNetwork(state_dim, action_dim, hidden_size).to(
            device=self.device
        )
        self.critic_target: QNetwork = QNetwork(state_dim, action_dim, hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            state_dim, action_dim, hidden_size
        ).to(self.device)

        # Entropy
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(cfg.env.action_dim).to(self.device)
            ).item()
            self.log_alpha: torch.Tensor = torch.zeros(
                1, requires_grad=True, device=self.device
            )
            self.alpha_optim: Adam = Adam([self.log_alpha], lr=cfg.lr)

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
