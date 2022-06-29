from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.optim import Adam

from mprl.models.sac_common import QNetwork
from mprl.utils.math_helper import hard_update

from .networks import GaussianMotionPrimitiveTimePolicy


class SACMP:
    def __init__(self, cfg: DictConfig):
        # Parameters
        self.gamma: float = cfg.gamma
        self.tau: float = cfg.tau
        self.alpha: float = cfg.alpha
        self.automatic_entropy_tuning: bool = cfg.automatic_entropy_tuning
        self.device: torch.device = torch.device(
            "cuda" if cfg.device == "cuda" else "cpu"
        )
        state_dim = cfg.env.state_dim
        action_dim = (cfg.num_basis + 1) * cfg.num_dof
        hidden_size = cfg.hidden_size

        # Networks
        self.critic = QNetwork(
            state_dim, action_dim, hidden_size, additional_actions=1
        ).to(device=self.device)
        self.critic_target = QNetwork(
            state_dim, action_dim, hidden_size, additional_actions=1
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy = GaussianMotionPrimitiveTimePolicy(
            state_dim,
            action_dim,
            hidden_size,
        ).to(self.device)

        # Entropy
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(cfg.env.action_dim).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

    def select_weights_and_time(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            weight_times, _, _, _ = self.policy.sample(state)
        else:
            _, _, weight_times, _ = self.policy.sample(state)
        return weight_times.squeeze(), {
            "time": weight_times.squeeze()[-1].detach().cpu().numpy(),
        }

    def sample(self, state):
        return self.policy.sample(state)

    def parameters(self):
        return self.policy.parameters()

    # Save model parameters
    def save(self, base_path, folder):
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

    # Load model parameters
    def load(self, path, evaluate=False):
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
