from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch as ch
import wandb
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.utils import SequenceRB
from mprl.utils.ds_helper import to_ts

from ...common.interfaces import Predictable
from ..mixed.agent import SACMixedMP
from ..mp_agent import SACMPBase
from .tr_networks import TrustRegionPolicy

LOG_PROB_MIN = -27.5
LOG_PROB_MAX = 0.0


class SACTRL(SACMixedMP):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        alpha_q: float,
        num_steps: int,
        lr: float,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        network_width: int,
        network_depth: int,
        action_scale: float,
        automatic_entropy_tuning: bool,
        kl_loss_scale: float,
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        planner_update: MPTrajectory,
        planner_imp_sampling: MPTrajectory,
        ctrl: Controller,
        buffer: SequenceRB,
        buffer_policy: SequenceRB,
        decompose_fn: Callable,
        model: Optional[Predictable] = None,
        policy_loss_type: str = "mean",  # other is "weighted"
        target_entropy: Optional[float] = None,
        layer_type: Optional[str] = "kl",
        mean_bound: Optional[float] = 1.0,
        cov_bound: Optional[float] = 1.0,
        use_imp_sampling: bool = False,
        learn_bc: bool = False,
    ):
        super().__init__(
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            automatic_entropy_tuning=automatic_entropy_tuning,
            alpha_q=alpha_q,
            num_steps=num_steps,
            lr=lr,
            batch_size=batch_size,
            state_dim=state_dim,
            action_dim=action_dim,
            num_basis=num_basis,
            network_width=network_width,
            network_depth=network_depth,
            action_scale=action_scale,
            planner_act=planner_act,
            planner_eval=planner_eval,
            planner_update=planner_update,
            planner_imp_sampling=planner_imp_sampling,
            ctrl=ctrl,
            buffer=buffer,
            decompose_fn=decompose_fn,
            model=model,
            policy_loss_type=policy_loss_type,
            target_entropy=target_entropy,
            use_imp_sampling=use_imp_sampling,
            q_loss="off_policy",
            action_clip=False,
            learn_bc=learn_bc,
            q_model_bc=None,
            buffer_policy=buffer_policy,
        )
        self.kl_loss_scaler = kl_loss_scale
        del self.policy
        del self.optimizer_policy

        self.policy = TrustRegionPolicy(
            (state_dim, (num_basis + 1) * self.num_dof),
            network_width,
            network_depth,
            layer_type=layer_type,
            mean_bound=mean_bound,
            cov_bound=cov_bound,
        )
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.policy.hard_update()

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
    def load(self, path):
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

    def parameters(self):
        return self.policy.parameters()

    def action_train(self, state: np.ndarray, info: any) -> np.ndarray:
        act = SACMPBase.action_train(self, state, info)
        (_, _), (mean, std) = self.policy.forward(to_ts(state[None]))
        self.c_weight_mean = mean[0].detach().cpu().numpy()
        self.c_weight_std = std[0].detach().cpu().numpy()
        return act

    def update_target_policy(self):
        self.policy.soft_update(self.tau)

    def _off_policy_mean_loss(self):
        # dimension (batch_size, sequence_len, weight_dimension)
        if self.buffer_policy is not None:
            buffer = self.buffer_policy
        else:
            buffer = self.buffer
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
            (des_qps, des_qvs),
            (next_des_qps, next_des_qvs),
            weight_means,
            weight_stds,
            idxs,
        ) = buffer.sample_batch(self.batch_size)

        p, proj_p = self.policy.forward(states[:, 0, :])
        if self.use_imp_sampling:
            weights, log_pi = self.policy.sample_log_prob_no_tanh(states.flatten(0, 1))
            weights = ch.reshape(weights, states.shape[:2] + (weights.shape[-1],))
            log_pi = ch.reshape(log_pi, states.shape[:2] + (1,))
        else:
            weights, log_pi = self.policy.sample_log_prob(states.flatten(0, 1))
            weights = ch.reshape(weights, states.shape[:2] + (weights.shape[-1],))
            log_pi = ch.reshape(log_pi, states.shape[:2] + (1,))
        self.planner_update.init(
            weights[:, 0, :],
            bc_pos=des_qps[:, 0, :],
            bc_vel=des_qvs[:, 0, :],
        )
        new_des_qps, new_des_qvs = self.planner_update.get_traj()
        b_q, b_v = self.decompose_fn(states, sim_states)
        new_actions = self.ctrl.get_action(
            new_des_qps[:, :-1, :], new_des_qvs[:, :-1, :], b_q, b_v
        )
        qf1_pi, qf2_pi = self.critic(states, new_actions)
        min_qf_pi = ch.min(qf1_pi, qf2_pi)
        kl_loss = self.policy.kl_regularization_loss(p, proj_p)
        policy_loss = (
            -min_qf_pi + self.alpha * log_pi
        ).mean() + self.kl_loss_scaler * kl_loss
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "kl_loss": kl_loss.detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def store(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        ch.save(
            {
                "policy_state_dict": self.policy.policy.state_dict(),
                "old_policy_state_dict": self.policy.old_policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "optimizer_critic_state_dict": self.optimizer_critic.state_dict(),
                "optimizer_policy_state_dict": self.optimizer_policy.state_dict(),
                **(
                    {"optimizer_entropy": self.optimizer_alpha.state_dict()}
                    if self.automatic_entropy_tuning
                    else {}
                ),
            },
            path + "/model.pt",
        )
        self.buffer.store(path + "/" + self.buffer.store_under())

    def load(self, path):
        ckpt_path = path + "/model.pt"
        if ckpt_path is not None:
            checkpoint = ch.load(ckpt_path)
            self.policy.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.policy.old_policy.load_state_dict(checkpoint["old_policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.optimizer_critic.load_state_dict(
                checkpoint["optimizer_critic_state_dict"]
            )
            self.optimizer_policy.load_state_dict(
                checkpoint["optimizer_policy_state_dict"]
            )
            if self.automatic_entropy_tuning:
                self.optimizer_alpha.load_state_dict(checkpoint["optimizer_entropy"])
            if self.model is not None:
                self.model.load_state_dict(checkpoint["model"])
        self.buffer.load(path + "/" + self.buffer.store_under())
        self.planner_act.reset_planner()
        self.planner_update.reset_planner()
        self.planner_eval.reset_planner()
        self.planner_imp_sampling.reset_planner()

    def store_under(self, path):
        return path + "sac-tr/"
