from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from torch.distributions import Independent, Normal
from torch.optim import Adam

from mprl.utils import SequenceRB
from mprl.utils.ds_helper import to_np, to_ts
from mprl.utils.math_helper import hard_update

from ...controllers import MPTrajectory, PDController
from .. import Actable, Evaluable, Serializable, Trainable
from ..common import QNetwork
from .networks import GaussianMotionPrimitivePolicy

LOG_PROB_MIN = -27.5
LOG_PROB_MAX = 0.0


class SACMixedMP(Actable, Trainable, Serializable, Evaluable):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        lr: float,
        batch_size: int,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        num_dof: int,
        network_width: int,
        network_depth: int,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        planner_update: MPTrajectory,
        ctrl: PDController,
        buffer: SequenceRB,
        decompose_fn: Callable,
    ):
        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.device: torch.device = device
        self.buffer = buffer
        self.planner_act: MPTrajectory = planner_act
        self.planner_eval: MPTrajectory = planner_eval
        self.planner_update: MPTrajectory = planner_update
        self.ctrl: PDController = ctrl
        self.decompose_fn = decompose_fn

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(device=self.device)
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianMotionPrimitivePolicy = GaussianMotionPrimitivePolicy(
            (state_dim, num_basis * num_dof), network_width, network_depth
        ).to(self.device)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)

    def sequence_reset(self):
        self.planner_act.reset_planner()

    @torch.no_grad()
    def action(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_act)
        except StopIteration:
            weights, _, _ = self.policy.sample(state)
            self.planner_act.re_init(
                weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
            )
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return to_np(action)

    def eval_reset(self) -> np.ndarray:
        self.planner_eval.reset_planner()

    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_act)
        except StopIteration:
            _, _, weights = self.policy.sample(state)
            self.planner_act.re_init(
                weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
            )
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return to_np(action)

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

    def sample(self, state, sim_state):
        self.planner_update.reset_planner()
        weight, logp, mean = self.policy.sample(state)
        b_q, b_v = self.decompose_fn(state, sim_state)
        self.planner_update.re_init(
            weight, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
        )
        q, v = next(self.planner_update)
        action, _ = self.ctrl.get_action(q, v, b_q, b_v)
        return to_ts(action), logp, mean

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
    def load(self, path):
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
        pass
