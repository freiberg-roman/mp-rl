from pathlib import Path
from random import randrange
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from mp_pytorch.util import tensor_linspace
from torch.optim import Adam

from mprl.controllers import Controller, MPTrajectory
from mprl.models import Actable, Evaluable, Predictable, Serializable, Trainable
from mprl.models.common import QNetwork
from mprl.models.common.policy_network import GaussianPolicy
from mprl.utils import SequenceRB
from mprl.utils.ds_helper import to_np, to_ts
from mprl.utils.math_helper import hard_update, soft_update

LOG_PROB_MIN = -27.5
LOG_PROB_MAX = 0.0


class SACMixedMP(Actable, Trainable, Serializable, Evaluable):
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
        device: torch.device,
        state_dim: int,
        action_dim: int,
        num_basis: int,
        num_dof: int,
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
        # Parameters
        self.gamma: float = gamma
        self.tau: float = tau
        self.alpha: float = alpha
        self.alpha_q: float = alpha_q
        self.automatic_entropy_tuning: bool = automatic_entropy_tuning
        self.target_entropy: Optional[float] = target_entropy
        self.num_steps: int = num_steps
        self.device: torch.device = device
        self.buffer: SequenceRB = buffer
        self.planner_act: MPTrajectory = planner_act
        self.planner_eval: MPTrajectory = planner_eval
        self.planner_update: MPTrajectory = planner_update
        self.planner_imp_sampling: MPTrajectory = planner_imp_sampling
        self.ctrl: Controller = ctrl
        self.decompose_fn = decompose_fn
        self.batch_size: int = batch_size
        self.mode: str = policy_loss_type
        self.model = model
        self.num_dof = num_dof
        self.use_imp_sampling = use_imp_sampling

        # Networks
        self.critic: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(device=self.device)
        self.critic_target: QNetwork = QNetwork(
            (state_dim, action_dim), network_width, network_depth
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy: GaussianPolicy = GaussianPolicy(
            (state_dim, (num_basis + 1) * num_dof),
            network_width,
            network_depth,
            action_scale=action_scale,
        ).to(self.device)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=lr)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=lr)
        if automatic_entropy_tuning:
            if self.target_entropy is None:
                self.target_entropy = -((num_basis + 1) * num_dof)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.optimizer_alpha = Adam([self.log_alpha], lr=lr)
        self.weights_log = []
        self.traj_log = []
        self.traj_des_log = []
        self.c_weight_mean = None
        self.c_weight_cov = None
        self.c_des_q = None
        self.c_des_v = None

    def sequence_reset(self):
        if len(self.buffer) > 0:
            self.buffer.close_trajectory()
        self.planner_act.reset_planner()

    @torch.no_grad()
    def action(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = torch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = torch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_act)
        except StopIteration:
            weights, _, _ = self.policy.sample(state)
            b_q_des, b_v_des = self.planner_act.get_next_bc()
            if b_q_des is not None and b_v_des is not None:
                self.planner_act.init(
                    weights,
                    bc_pos=b_q_des[None],
                    bc_vel=b_v_des[None],
                    num_t=self.num_steps,
                )
            else:
                self.planner_act.init(
                    weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
                )
            q, v = next(self.planner_act)

        if self.use_imp_sampling:
            (self.c_weight_mean, self.c_weight_cov) = self.policy.forward(state)
            self.c_weight_cov = self.c_weight_cov.exp()
        self.c_des_q = q
        self.c_des_v = v
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return to_np(action.squeeze())

    def eval_reset(self) -> np.ndarray:
        self.planner_eval.reset_planner()
        self.weights_log = []
        self.traj_log = []
        self.traj_des_log = []

    def eval_log(self) -> Dict:
        times_full_traj = tensor_linspace(
            0.0, self.planner_eval.dt * len(self.traj_log), len(self.traj_log)
        )
        # just last planned trajectory
        times = tensor_linspace(
            0,
            self.planner_eval.dt * self.planner_eval.num_t,
            self.planner_eval.num_t + 1,
        )
        pos = self.planner_eval.current_traj
        if pos is None:
            return {}
        plt.close("all")
        figs = {}

        for dim in range(self.num_dof):
            figs["last_desired_traj"] = plt.figure()
            ax = figs["last_desired_traj"].add_subplot(111)
            ax.plot(times, pos[:, dim])

        pos_real = np.array(self.traj_log)
        pos_des = np.array(self.traj_des_log)
        for dim in range(self.num_dof):
            figs["traj_" + str(dim)] = plt.figure()
            ax = figs["traj_" + str(dim)].add_subplot(111)
            ax.plot(times_full_traj, pos_real[:, dim])
            ax.plot(times_full_traj, pos_des[:, dim])

        pos_delta = np.abs(pos_real - pos_des)
        plt.figure()
        for dim in range(self.num_dof):
            figs["abs_delta_traj_" + str(dim)] = plt.figure()
            ax = figs["abs_delta_traj_" + str(dim)].add_subplot(111)
            ax.plot(times_full_traj, pos_delta[:, dim])
        return {
            **figs,
            "weights_histogram": wandb.Histogram(np.array(self.weights_log).flatten()),
        }

    @torch.no_grad()
    def action_eval(self, state: np.ndarray, info: any) -> np.ndarray:
        sim_state = info
        b_q, b_v = self.decompose_fn(state, sim_state)
        b_q = torch.FloatTensor(b_q).to(self.device).unsqueeze(0)
        b_v = torch.FloatTensor(b_v).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        try:
            q, v = next(self.planner_eval)
        except StopIteration:
            _, _, weights = self.policy.sample(state)
            b_q_des, b_v_des = self.planner_eval.get_next_bc()
            if b_q_des is not None and b_v_des is not None:
                self.planner_eval.init(
                    weights,
                    bc_pos=b_q_des[None],
                    bc_vel=b_v_des[None],
                    num_t=self.num_steps,
                )
            else:
                self.planner_eval.init(
                    weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps
                )
            q, v = next(self.planner_eval)
            self.weights_log.append(to_np(weights.squeeze()).flatten())
        action = self.ctrl.get_action(q, v, b_q, b_v)
        q_curr = self.planner_eval.get_current()
        self.traj_des_log.append(to_np((q_curr)))
        self.traj_log.append(to_np(b_q[0]))
        return to_np(action.squeeze())

    def add_step(
        self,
        state: np.ndarray,
        next_state: np.array,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        if self.use_imp_sampling:
            self.buffer.add(
                state,
                next_state,
                action,
                reward,
                done,
                sim_state,
                weight_mean=self.c_weight_mean,
                weight_cov=self.c_weight_cov,
                des_q=self.c_des_q,
                des_v=self.c_des_v,
            )
        else:
            self.buffer.add(
                state,
                next_state,
                action,
                reward,
                done,
                sim_state,
                des_q=self.c_des_q,
                des_v=self.c_des_v,
            )

    def sample(self, states, sim_states):
        self.planner_update.reset_planner()
        weights, logp, mean = self.policy.sample(states)
        b_q, b_v = self.decompose_fn(states, sim_states)
        self.planner_update.init(weights, bc_pos=b_q, bc_vel=b_v, num_t=self.num_steps)
        q, v = next(self.planner_update)
        action = self.ctrl.get_action(q, v, b_q, b_v)
        return action, logp, mean

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
            _,
            _,
        ) = batch.to_torch_batch()

        # Compute critic loss
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.sample(
                next_states, sim_states
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha_q * next_state_log_pi
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
        return {
            **{
                "critic_loss": qf_loss.item(),
                "policy_loss": loss.item(),
                "alpha": self.alpha,
            },
            **loggable,
            **model_loss,
        }

    def _off_policy_mean_loss(self):
        batch = next(
            self.buffer.get_true_k_sequence_iter(1, self.num_steps, self.batch_size)
        )
        # dimensions (batch_size, sequence_len, data_dimension)
        if self.use_imp_sampling:
            (
                states,
                next_states,
                actions,
                rewards,
                dones,
                sim_states,
                des_qs,
                des_vs,
                weight_means,
                weight_covs,
            ) = batch.to_torch_batch()
        else:
            (
                states,
                next_states,
                actions,
                rewards,
                dones,
                sim_states,
                des_qs,
                des_vs,
            ) = batch.to_torch_batch()
        # dimension (batch_size, sequence_len, weight_dimension)
        weights, log_pi, _ = self.policy.sample(states[:, 0, :])
        if self.use_imp_sampling:
            mean, log_std = self.policy.forward(states[:, 0, :])
            std = log_std.exp()
        b_q, b_v = self.decompose_fn(states, sim_states)
        self.planner_update.init(
            weights,
            bc_pos=des_qs[:, 0, :],
            bc_vel=des_vs[:, 0, :],
            num_t=self.num_steps,
        )
        next_s = states[:, 0, :]
        next_sim_states = sim_states[0][:, 0, :], sim_states[1][:, 0, :]
        min_qf = 0
        for i, qv in enumerate(self.planner_update):
            q, v = qv
            b_q, b_v = self.decompose_fn(next_s, next_sim_states)
            action = self.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = self.critic(next_s, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            # we simply take the sequence as fixed
            if i == self.num_steps - 1:
                break
            next_s, next_sim_states = next_states[:, i, :], (
                sim_states[0][:, i + 1, :],
                sim_states[1][:, i + 1, :],
            )
        min_qf /= self.num_steps
        if self.use_imp_sampling:
            with torch.no_grad():
                traj_old_des = torch.concat(
                    (to_ts(b_q).unsqueeze(dim=1), des_qs), dim=1
                )
                old_log_prob = self.planner_imp_sampling.get_log_prob(
                    traj_old_des,
                    weight_means[:, 0, :],
                    weight_covs[:, 0, :],
                    to_ts(b_q),
                    to_ts(b_v),
                )
                curr_log_prob = self.planner_imp_sampling.get_log_prob(
                    traj_old_des, mean, std, to_ts(b_q), to_ts(b_v)
                )
                imp = (curr_log_prob - old_log_prob).clamp(max=0.0).exp()
                to_add = {
                    "imp_mean": imp.detach().cpu().mean().item(),
                }
            policy_loss = (imp * ((-min_qf) + self.alpha * log_pi)).mean()
        else:
            policy_loss = (-min_qf).mean() + self.alpha * log_pi.mean()
            to_add = {}
        return policy_loss, {
            **to_add,
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weight_goal_max": weights[..., -1].detach().cpu().max().item(),
            "weight_goal_min": weights[..., -1].detach().cpu().min().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _model_policy_mean_loss(self):
        batch = next(self.buffer.get_iter(1, self.batch_size))
        # dimension (batch_size, data_dimension)
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
            des_qs,
            des_vs,
        ) = batch.to_torch_batch()
        # dimension (batch_size, weight_dimension)
        weights, log_pi, _ = self.policy.sample(states)
        b_q, b_v = self.decompose_fn(states, sim_states)
        self.planner_update.init(
            weights, bc_pos=des_qs, bc_vel=des_vs, num_t=self.num_steps
        )
        next_states = states
        next_sim_states = sim_states
        min_qf = 0
        for q, v in self.planner_update:
            b_q, b_v = self.decompose_fn(next_states, next_sim_states)
            action = self.ctrl.get_action(q, v, b_q, b_v)

            # compute q val: dimension (batch_size, 1)
            qf1_pi, qf2_pi = self.critic(next_states, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            next_states, next_sim_states = self.model.next_state(
                next_states, action, next_sim_states
            )
            next_states = to_ts(next_states).to(self.device)
        min_qf /= self.num_steps
        policy_loss = (-min_qf).mean() + self.alpha * log_pi.mean()
        if isinstance(self.model, Trainable):
            self.model.update(batch=batch)
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weight_goal_max": weights[..., -1].detach().cpu().max().item(),
            "weight_goal_min": weights[..., -1].detach().cpu().min().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _off_policy_mean_performance_loss(self):
        batch = next(
            self.buffer.get_true_k_sequence_iter(1, self.num_steps, self.batch_size)
        )
        # dimensions (batch_size, sequence_len, data_dimension)
        if self.use_imp_sampling:
            (
                states,
                next_states,
                actions,
                rewards,
                dones,
                sim_states,
                des_qs,
                des_vs,
                weight_means,
                weight_covs,
            ) = batch.to_torch_batch()
        else:
            (
                states,
                next_states,
                actions,
                rewards,
                dones,
                sim_states,
                des_qs,
                des_vs,
            ) = batch.to_torch_batch()
        # dimension (batch_size, sequence_len, weight_dimension)
        weights, log_pi, _ = self.policy.sample(states[:, 0, :])
        if self.use_imp_sampling:
            (mean, log_std) = self.policy.forward(states[:, 0, :])
            std = log_std.exp()
        b_q, b_v = self.decompose_fn(states, sim_states)
        self.planner_update.init(
            weights,
            bc_pos=des_qs[:, 0, :],
            bc_vel=des_vs[:, 0, :],
            num_t=self.num_steps,
        )
        # Compute loss for one random time step in the sequence
        i = randrange(self.num_steps)
        q, v = self.planner_update[i]
        s = states[:, i, :]
        sim_s = sim_states[0][:, i, :], sim_states[1][:, i, :]
        b_q, b_v = self.decompose_fn(s, sim_s)
        a = self.ctrl.get_action(q, v, b_q, b_v)
        qf1_pi, qf2_pi = self.critic(s, a)
        min_qf = torch.min(qf1_pi, qf2_pi)
        if self.use_imp_sampling:
            with torch.no_grad():
                traj_old_des = torch.concat(
                    (to_ts(b_q).unsqueeze(dim=1), des_qs), dim=1
                )
                old_log_prob = self.planner_imp_sampling.get_log_prob(
                    traj_old_des,
                    weight_means[:, 0, :],
                    weight_covs[:, 0, :],
                    to_ts(b_q),
                    to_ts(b_v),
                )
                curr_log_prob = self.planner_imp_sampling.get_log_prob(
                    traj_old_des, mean, std, to_ts(b_q), to_ts(b_v)
                )
                imp = (curr_log_prob - old_log_prob).clamp(max=0.0).exp()
                to_add = {
                    "imp_mean": imp.detach().cpu().mean().item(),
                }
            policy_loss = (imp * ((-min_qf) + self.alpha * log_pi)).mean()
        else:
            policy_loss = (-min_qf).mean() + self.alpha * log_pi.mean()
            to_add = {}
        return policy_loss, {
            **to_add,
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_mean": weights[..., :-1].detach().cpu().mean().item(),
            "weight_std": weights[..., :-1].detach().cpu().std().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weight_goal_max": weights[..., -1].detach().cpu().max().item(),
            "weight_goal_min": weights[..., -1].detach().cpu().min().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }

    def _model_mean_performance_loss(self):
        batch = next(self.buffer.get_iter(1, self.batch_size))
        # dimension (batch_size, data_dimension)
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
            des_qs,
            des_vs,
        ) = batch.to_torch_batch()
        # dimension (batch_size, weight_dimension)
        weights, log_pi, _ = self.policy.sample(states)
        b_q, b_v = self.decompose_fn(states, sim_states)
        self.planner_update.init(
            weights, bc_pos=des_qs, bc_vel=des_vs, num_t=self.num_steps
        )
        next_states = states
        next_sim_states = sim_states
        loss_at_iter = randrange(self.num_steps)
        for i, qv in enumerate(self.planner_update):
            q, v = qv
            b_q, b_v = self.decompose_fn(next_states, next_sim_states)
            action = self.ctrl.get_action(q, v, b_q, b_v)

            # compute q val: dimension (batch_size, 1)
            if loss_at_iter == i:
                qf1_pi, qf2_pi = self.critic(next_states, action)
                min_qf = torch.min(qf1_pi, qf2_pi)
                break
            next_states, next_sim_states = self.model.next_state(
                next_states, action, next_sim_states
            )
            next_states = to_ts(next_states).to(self.device)
        policy_loss = (-min_qf).mean() + self.alpha * log_pi.mean()
        if isinstance(self.model, Trainable):
            self.model.update(batch=batch)
        return policy_loss, {
            "entropy": (-log_pi).detach().cpu().mean().item(),
            "weight_goal_mean": weights[..., -1].detach().cpu().mean().item(),
            "weight_goal_std": weights[..., -1].detach().cpu().std().item(),
            "weight_goal_max": weights[..., -1].detach().cpu().max().item(),
            "weight_goal_min": weights[..., -1].detach().cpu().min().item(),
            "weights_histogram": wandb.Histogram(
                weights[..., :-1].detach().cpu().numpy().flatten()
            ),
        }
