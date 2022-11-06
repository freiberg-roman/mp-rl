from pathlib import Path
from random import randrange
from typing import Callable, Optional, Tuple

import numpy as np
import torch as ch
import torch.nn.functional as F
import wandb
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.utils import SequenceRB
from mprl.utils.ds_helper import to_ts
from mprl.utils.math_helper import soft_update

from ...common import Predictable, Trainable
from ...common.policy_network import GaussianPolicy
from ..mp_agent import SACMPBase

LOG_PROB_MIN = -27.5
LOG_PROB_MAX = 0.0


class SACMixedMP(SACMPBase):
    def __init__(
        self,
        gamma: float,
        tau: float,
        alpha: float,
        automatic_entropy_tuning: bool,
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
        planner_act: MPTrajectory,
        planner_eval: MPTrajectory,
        planner_update: MPTrajectory,
        planner_imp_sampling: MPTrajectory,
        ctrl: Controller,
        buffer: SequenceRB,
        decompose_fn: Callable,
        model: Optional[Predictable] = None,
        policy_loss_type: str = "mean",  # other is "weighted"
        target_entropy: Optional[float] = None,
        use_imp_sampling: bool = False,
    ):
        super().__init__(
            action_dim,
            ctrl,
            decompose_fn,
            lr,
            network_width,
            network_depth,
            planner_act,
            planner_eval,
            state_dim,
        )

        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.alpha_q: float = alpha_q
        self.automatic_entropy_tuning: bool = automatic_entropy_tuning
        self.target_entropy: Optional[float] = target_entropy
        self.num_steps: int = num_steps
        self.buffer: SequenceRB = buffer
        self.planner_update: MPTrajectory = planner_update
        self.planner_imp_sampling: MPTrajectory = planner_imp_sampling
        self.ctrl: Controller = ctrl
        self.decompose_fn = decompose_fn
        self.batch_size: int = batch_size
        self.mode: str = policy_loss_type
        self.model = model
        self.use_imp_sampling = use_imp_sampling

        # Networks
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, (num_basis + 1) * self.num_dof),
            network_width,
            network_depth,
            action_scale=action_scale,
        )
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        if automatic_entropy_tuning:
            if self.target_entropy is None:
                self.target_entropy = -((num_basis + 1) * self.num_dof)
            self.log_alpha = ch.zeros(1, requires_grad=True)
            self.optimizer_alpha = Adam([self.log_alpha], lr=lr)

        self.c_weight_mean = None
        self.c_weight_std = None

    def sequence_reset(self):
        if len(self.buffer) > 0:
            self.buffer.close_trajectory()
        self.planner_act.reset_planner()

    def sample(self, state: ch.Tensor) -> ch.Tensor:
        if self.use_imp_sampling:
            return self.policy.sample_no_tanh(state)
        else:
            return self.policy.sample(state)

    def action_train(self, state: np.ndarray, info: any) -> np.ndarray:
        act = super().action_train(state, info)
        mean, std_log = self.policy.forward(to_ts(state[None]))
        self.c_weight_mean = mean.detach().cpu().numpy()
        self.c_weight_std = std_log.exp().detach().cpu().numpy()
        return act

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        self.buffer.add(
            state,
            next_state,
            action,
            reward,
            done,
            sim_state,
            (self.c_des_q, self.c_des_v),
            (self.c_des_q_next, self.c_des_v_next),
            weight_mean=self.c_weight_mean,
            weight_std=self.c_weight_std,
        )

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

    def update(self) -> dict:
        batch = next(self.buffer.get_iter(1, self.batch_size))
        if self.model is not None and isinstance(self.model, Trainable):
            model_loss = self.model.update(batch=batch)
        else:
            model_loss = {}
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
            (des_qps, des_qvs),
            (_, _),
            weight_means,
            weight_stds,
            _,
        ) = self.buffer.sample_batch(self.batch_size, sequence=False)

        # Compute critic loss
        with ch.no_grad():
            # Compute next action
            if self.use_imp_sampling:
                weights = self.policy.sample_no_tanh(states)
                _, weights_log_pi_next = self.policy.sample_log_prob_no_tanh(
                    next_states
                )
            else:
                weights = self.policy.sample(states)
                _, weights_log_pi_next = self.policy.sample_log_prob(next_states)
            self.planner_update.init(weights, bc_pos=des_qps, bc_vel=des_qvs)
            next_q, next_v = self.planner_update[1]
            b_next_q, b_next_v = self.decompose_fn(next_states, None)
            next_state_action = self.ctrl.get_action(next_q, next_v, b_next_q, b_next_v)

            qf1_next_target, qf2_next_target = self.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                ch.min(qf1_next_target, qf2_next_target)
                - self.alpha_q * weights_log_pi_next
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

        if self.model is None:
            if self.mode == "mean":
                loss, loggable = self._off_policy_mean_loss()
            elif self.mode == "mean_performance":
                loss, loggable = self._off_policy_mean_performance_loss()
            else:
                raise ValueError("Invalid mode")
        else:
            if self.mode == "mean":
                loss, loggable = self._model_policy_mean_loss()
            elif self.mode == "mean_performance":
                loss, loggable = self._model_mean_performance_loss()
            else:
                raise ValueError("Invalid mode")

        # Update policy
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (-loggable["entropy"] + self.target_entropy)
            ).mean()
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()
            self.alpha = self.log_alpha.exp()
            loggable["alpha_loss"] = alpha_loss.item()

        soft_update(self.critic_target, self.critic, self.tau)
        self.update_target_policy()
        return {
            **{
                "critic_loss": qf_loss.item(),
                "policy_loss": loss.item(),
                "alpha": self.alpha,
            },
            **loggable,
            **model_loss,
        }

    def importance_weights(self, traj, target_dist, behavior_dist, bc_q, bc_v):
        target_mean, target_std = target_dist
        behavior_mean, behavior_std = behavior_dist
        assert np.allclose(bc_q, traj[:, 0, :])
        old_log_prob = self.planner_imp_sampling.get_log_prob(
            traj,
            behavior_mean,
            behavior_std,
            bc_q,
            bc_v,
        )
        cur_log_prob = self.planner_imp_sampling.get_log_prob(
            traj,
            target_mean,
            target_std,
            bc_q,
            bc_v,
        )
        imp = (cur_log_prob - old_log_prob).clamp(max=0.0).exp()
        imp = ch.nn.functional.normalize(imp, p=1, dim=-1) * len(imp)
        to_add = {
            "importance_weights": wandb.Histogram(imp.detach().cpu().numpy().flatten()),
        }
        return imp, to_add

    def update_target_policy(self):
        pass

    def _off_policy_mean_loss(self):
        # dimensions (batch_size, sequence_len, data_dimension)
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
        ) = self.buffer.sample_batch(self.batch_size)
        # dimension (batch_size, sequence_len, weight_dimension)
        if self.use_imp_sampling:
            weights, log_pi = self.policy.sample_log_prob_no_tanh(states)
            mean, log_std = self.policy.forward(states[:, 0, :])
            std = log_std.exp()
        else:
            weights, log_pi = self.policy.sample_log_prob(states)
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
        if self.use_imp_sampling:
            traj_old_des = ch.concat(
                (des_qps, next_des_qps[:, -1, :].unsqueeze(dim=1)), dim=1
            )
            with ch.no_grad():
                imp, to_add = self.importance_weights(
                    traj_old_des,
                    (mean, std),
                    (weight_means[:, 0, :], weight_stds[:, 0, :]),
                    des_qps[:, 0, :],
                    des_qvs[:, 0, :],
                )
            policy_loss = (imp * (-min_qf_pi + self.alpha * log_pi)).mean()
        else:
            policy_loss = (-min_qf_pi + self.alpha * log_pi).mean()
            # self.buffer.update_des_qvs(idxs, new_des_qps, new_des_qvs)
            to_add = {}
        return policy_loss, {
            **to_add,
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _off_policy_mean_performance_loss(self):
        """No importance sampling version here"""
        # dimensions (batch_size, sequence_len, data_dimension)
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
        ) = self.buffer.sample_batch(self.batch_size)
        # dimension (batch_size, sequence_len, weight_dimension)
        weights, log_pi = self.policy.sample_log_prob(states)
        self.planner_update.init(
            weights[:, 0, :],
            bc_pos=des_qps[:, 0, :],
            bc_vel=des_qvs[:, 0, :],
        )
        loss_at_iter = randrange(self.num_steps)
        new_des_qps, new_des_qvs = self.planner_update[loss_at_iter]
        b_q, b_v = self.decompose_fn(states[:, loss_at_iter, :], None)
        new_actions = self.ctrl.get_action(new_des_qps, new_des_qvs, b_q, b_v)
        qf1_pi, qf2_pi = self.critic(states[:, loss_at_iter, :], new_actions)
        min_qf_pi = ch.min(qf1_pi, qf2_pi)
        policy_loss = (self.gamma**loss_at_iter) * (
            -min_qf_pi + self.alpha * log_pi[:, loss_at_iter, :]
        ).mean()
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _model_policy_mean_loss(self):
        # dimensions (batch_size, sequence_len, data_dimension)
        (
            states,
            _,
            actions,
            rewards,
            dones,
            sim_states,
            (des_qps, des_qvs),
            (_, _),
            weight_means,
            weight_covs,
            idxs,
        ) = self.buffer.sample_batch(self.batch_size, sequence=False)
        # dimension (batch_size, weight_dimension)
        weights, log_pi = self.policy.sample_log_prob(states)
        self.planner_update.init(weights, bc_pos=des_qps, bc_vel=des_qvs)
        c_s = states
        c_s_sim = sim_states
        min_qf = 0
        for des_qp, des_qv in self.planner_update:
            c_bq, c_bv = self.decompose_fn(c_s, c_s_sim)
            action = self.ctrl.get_action(des_qp, des_qv, c_bq, c_bv)

            # compute q val: dimension (batch_size, 1)
            qf1_pi, qf2_pi = self.critic(c_s, action)
            min_qf += ch.min(qf1_pi, qf2_pi)
            c_s, c_s_sim = self.model.next_state(c_s, action, c_s_sim)
            c_s = to_ts(c_s)
            _, log_pi_c_s = self.policy.sample_log_prob(c_s)
            log_pi += log_pi_c_s
        min_qf /= self.num_steps
        log_pi /= self.num_steps
        policy_loss = (-min_qf + self.alpha * log_pi).mean()
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _model_mean_performance_loss(self):
        (
            states,
            _,
            actions,
            rewards,
            dones,
            sim_states,
            (des_qps, des_qvs),
            (_, _),
            weight_means,
            weight_covs,
            idxs,
        ) = self.buffer.sample_batch(self.batch_size, sequence=False)
        # dimension (batch_size, weight_dimension)
        weights, log_pi = self.policy.sample_log_prob(states)
        self.planner_update.init(weights, bc_pos=des_qps, bc_vel=des_qvs)
        c_s = states
        c_s_sim = sim_states
        min_qf = 0
        loss_at_iter = randrange(self.num_steps)
        for i, (des_qp, des_qv) in enumerate(self.planner_update):
            c_bq, c_bv = self.decompose_fn(c_s, c_s_sim)
            action = self.ctrl.get_action(des_qp, des_qv, c_bq, c_bv)

            # compute q val: dimension (batch_size, 1)
            if loss_at_iter == i:
                qf1_pi, qf2_pi = self.critic(c_s, action)
                min_qf = ch.min(qf1_pi, qf2_pi)
                c_s, c_s_sim = self.model.next_state(c_s, action, c_s_sim)
                c_s = to_ts(c_s)
                _, log_pi_c_s = self.policy.sample_log_prob(c_s)
                break
            c_s, c_s_sim = self.model.next_state(c_s, action, c_s_sim)
            c_s = to_ts(c_s)
        policy_loss = (-min_qf + self.alpha * log_pi).mean()
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def store_under(self, path):
        return "sac-mixed-mp"

    def store(self, path):
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

    def load(self, path):
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
        self.planner_act.reset_planner()
        self.planner_update.reset_planner()
        self.planner_eval.reset_planner()
        self.planner_imp_sampling.reset_planner()
