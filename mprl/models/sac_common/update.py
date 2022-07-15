from typing import Union

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.models.physic.prediction import Prediction
from mprl.utils import EnvSteps
from mprl.utils.math_helper import soft_update


class SACUpdate:
    def __init__(self):
        self.times_called = 0

    def __call__(
        self,
        agent: any,
        optimizer_policy: Adam,
        optimizer_critic: Adam,
        batch: EnvSteps,
    ) -> dict:
        # Sample a batch from memory
        states, next_states, actions, rewards, dones = batch.to_torch_batch()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = agent.sample(next_states)
            qf1_next_target, qf2_next_target = agent.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - agent.alpha * next_state_log_pi
            )
            next_q_value = rewards + (1 - dones.to(torch.float32)) * agent.gamma * (
                min_qf_next_target
            )
        qf1, qf2 = agent.critic(
            states, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        optimizer_critic.zero_grad()
        qf_loss.backward()
        optimizer_critic.step()
        pi, log_pi, _, _ = agent.sample(states)
        qf1_pi, qf2_pi = agent.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (agent.alpha * log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        if agent.automatic_entropy_tuning:
            alpha_loss = -(
                agent.log_alpha * (log_pi + agent.target_entropy).detach()
            ).mean()

            agent.alpha_optim.zero_grad()
            alpha_loss.backward()
            agent.alpha_optim.step()

            agent.alpha = agent.log_alpha.exp()

        soft_update(agent.critic_target, agent.critic, agent.tau)

        self.times_called += 1
        return {
            "sac_update_times_called": self.times_called,
            "qf1_loss": qf_loss.item(),
            "qf2_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
        }


class SACModelUpdate:
    def __init__(self, model: Prediction):
        self.times_called: int = 0
        self.model: Prediction = model

    def __call__(
            self,
            agent: any,
            optimizer_policy: Adam,
            optimizer_critic: Adam,
            batch: EnvSteps,
    ) -> dict:
        # Sample a batch from memory
        states, next_states, actions, rewards, dones = batch.to_torch_batch()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = agent.sample(next_states)
            qf1_next_target, qf2_next_target = agent.critic_target(
                next_states, next_state_action
            )
            min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - agent.alpha * next_state_log_pi
            )
            next_q_value = rewards + (1 - dones.to(torch.float32)) * agent.gamma * (
                min_qf_next_target
            )
        qf1, qf2 = agent.critic(
            states, actions
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        optimizer_critic.zero_grad()
        qf_loss.backward()
        optimizer_critic.step()

        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        agent.planner_train.re_init(weights[0], bc_pos=b_q, bc_vel=b_v, num_t=agent.num_steps + 1)
        next_states = states
        min_qf = 0
        called = 0
        for q, v in agent.planner_train:
            b_q, b_v = agent.decompose_fn(next_states)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(states, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            next_states = self.model.next_state(next_states, action)
            called += 1
        min_qf /= called


        policy_loss = (-min_qf).mean()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
        soft_update(agent.critic_target, agent.critic, agent.tau)

        self.times_called += 1
        return {
            "sac_update_times_called": self.times_called,
            "qf1_loss": qf_loss.item(),
            "qf2_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
        }
