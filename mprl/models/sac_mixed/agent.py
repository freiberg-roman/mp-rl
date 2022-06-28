from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam

from mprl.models.sac_common import QNetwork
from mprl.utils.ds_helper import to_ts
from mprl.utils.math_helper import hard_update

from .networks import GaussianMotionPrimitivePolicy


class SACMixed:
    def __init__(
        self,
        cfg: OmegaConf,
        planner,
        ctrl,
        decompose_state_fn: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ):
        # Parameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.alpha = cfg.alpha
        self.automatic_entropy_tuning = cfg.automatic_entropy_tuning
        self.device = torch.device("cuda" if cfg.device == "cuda" else "cpu")
        state_dim = cfg.env.state_dim
        action_dim = cfg.env.action_dim
        hidden_size = cfg.hidden_size
        self.planner = planner
        self.ctrl = ctrl
        self.decompose_fn = decompose_state_fn

        # Networks
        self.critic = QNetwork(state_dim, action_dim, hidden_size, use_time=False).to(
            device=self.device
        )
        self.critic_target = QNetwork(
            state_dim, action_dim, hidden_size, use_time=False
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy = GaussianMotionPrimitivePolicy(
            state_dim,
            (cfg.num_basis + 1) * cfg.num_dof,
            hidden_size,
            full_std=cfg.full_std,
            learn_time=False,
        ).to(self.device)

        # Entropy
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(cfg.env.action_dim).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

    def select_action(self, state, bias=None, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            weight, _, _ = self.policy.sample(state)
        else:
            _, _, weight = self.policy.sample(state)
        b_q, b_v = self.decompose_fn(state)
        q, v = self.planner.one_step_ctrl(weight, b_q, b_v)
        action = self.ctrl.get_action(q, v, b_q, b_v, bias=bias)
        return action

    def select_weights_and_time(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            weight_times, _, _ = self.policy.sample(state)
        else:
            _, _, weight_times = self.policy.sample(state)
        return weight_times.squeeze()

    def sample(self, state, bias=None):
        weight, logp, mean = self.policy.sample(state)
        b_q, b_v = self.decompose_fn(state)
        q, v = self.planner.one_step_ctrl(weight, b_q, b_v)
        action = self.ctrl.get_action(q, v, b_q, b_v, bias=bias)
        return to_ts(action), logp, mean

    def parameters(self):
        return self.policy.parameters()

    # Save model parameters
    def save(self, base_path, folder):
        path = base_path + folder + "/mix-sac/"
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
        ckpt_path = path + "/mix-sac/model.pt"
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
