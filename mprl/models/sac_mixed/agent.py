from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Independent, Normal
from torch.optim import Adam

from mprl.models.sac_common import QNetwork
from mprl.utils.ds_helper import to_ts
from mprl.utils.math_helper import hard_update

from ...controllers import MPTrajectory, PDController
from .networks import GaussianMotionPrimitivePolicy

LOG_PROB_MIN = -27.5
LOG_PROB_MAX = 0.0


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
        self.planner: MPTrajectory = planner
        self.planner_train: MPTrajectory = deepcopy(planner)
        self.ctrl: PDController = ctrl
        self.decompose_fn: Callable[[np.ndarray], np.ndarray] = decompose_state_fn
        self.num_steps: int = cfg.time_steps
        self.force_replan: bool = False

        # Networks
        self.critic: QNetwork = QNetwork(state_dim, action_dim, hidden_size).to(
            device=self.device
        )
        self.critic_target: QNetwork = QNetwork(state_dim, action_dim, hidden_size).to(
            self.device
        )
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianMotionPrimitivePolicy = GaussianMotionPrimitivePolicy(
            state_dim,
            (cfg.num_basis + 1) * cfg.num_dof,
            hidden_size,
        ).to(self.device)

        # Entropy
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(cfg.env.action_dim).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=cfg.lr)

    @torch.no_grad()
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        b_q, b_v = self.decompose_fn(state)
        try:
            if self.force_replan:
                self.force_replan = False
                raise StopIteration

            q, v = next(self.planner)
        except StopIteration:
            if not evaluate:
                weight, _, _, _ = self.policy.sample(state)
            else:
                _, _, weight, _ = self.policy.sample(state)
            self.planner.re_init(weight, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps)
            q, v = next(self.planner)
        action, info = self.ctrl.get_action(q, v, b_q, b_v)
        return action, info

    def select_weights_and_time(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            weight_times, _, _, info = self.policy.sample(state)
        else:
            _, _, weight_times, info = self.policy.sample(state)
        return weight_times, info

    def sample(self, state, sim_state):
        weight, logp, mean, _ = self.policy.sample(state)
        b_q, b_v = self.decompose_fn(sim_state)
        self.planner_train.re_init(weight, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps)
        q, v = next(self.planner_train)
        action, _ = self.ctrl.get_action(q, v, b_q, b_v)
        return to_ts(action), logp, mean, {}

    def prob(self, states, weights):
        """
        expected dimensions of states: [num_samples, state_dim]
        expected dimensions of weights: [num_samples, weight_dim]

        returns: [num_samples, 1]
        """
        mean, log_std = self.policy.forward(states)
        std = log_std.exp()
        normal_dist = Independent(Normal(mean, std), 1)
        log_prob = torch.clamp(
            normal_dist.log_prob(weights), min=LOG_PROB_MIN, max=LOG_PROB_MAX
        )
        prob = log_prob.exp().unsqueeze(1)
        return prob

    def replan(self):
        self.force_replan = True

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
