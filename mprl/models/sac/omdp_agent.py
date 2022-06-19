from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import Adam

from mprl.models.sac.networks import GaussianMPTimePolicy, QNetwork
from mprl.utils.math_helper import hard_update


class OMDPSAC:
    def __init__(self, cfg, use_wandb=False):
        # Parameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.device = torch.device("cuda" if cfg.device == "cuda" else "cpu")
        self.learn_time = cfg.learn_time
        if not self.learn_time:
            self.time_steps = cfg.time_steps
        state_dim = cfg.env.state_dim
        action_dim = (cfg.num_basis + 1) * cfg.num_dof
        hidden_size = cfg.hidden_size
        self.use_wandb = use_wandb

        # Networks
        self.critic = QNetwork(
            state_dim, action_dim, hidden_size, use_time=self.learn_time
        ).to(device=self.device)
        self.critic_target = QNetwork(
            state_dim, action_dim, hidden_size, use_time=self.learn_time
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy = GaussianMPTimePolicy(
            state_dim,
            action_dim,
            hidden_size,
            full_std=cfg.full_std,
            learn_time=self.learn_time,
        ).to(self.device)
        if self.use_wandb:
            wandb.watch(self.critic, log="all")
            wandb.watch(self.policy, log="all")

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
            weight_times, _, _ = self.policy.sample(state)
        else:
            _, _, weight_times = self.policy.sample(state)
        if self.use_wandb:
            hist_value = weight_times.squeeze().detach().cpu().numpy()
            wandb.log(
                {"net_weights": wandb.Histogram(np_histogram=np.histogram(hist_value))}
            )
        return weight_times.squeeze()

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
