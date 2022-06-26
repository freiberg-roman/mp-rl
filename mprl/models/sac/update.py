import torch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.models.sac.mdp_agent import MDPSAC
from mprl.utils import EnvSteps
from mprl.utils.math_helper import soft_update


class SacUpdate:
    def __init__(self):
        self.times_called = 0

    def __call__(
        self,
        agent: MDPSAC,
        optimizer_policy: Adam,
        optimizer_critic: Adam,
        batch: EnvSteps,
    ):
        # Sample a batch from memory
        states, next_states, actions, rewards, dones = batch.to_torch_batch()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = agent.sample(next_states)
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
        pi, log_pi, _ = agent.sample(states)
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
