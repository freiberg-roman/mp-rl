import torch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.models.sac.mdp_agent import MDPSAC
from mprl.utils import EnvSteps
from mprl.utils.math_helper import soft_update


def train_mdp_sac(
    agent: MDPSAC,
    optimizer_policy: Adam,
    optimizer_critic: Adam,
    batch: EnvSteps,
    use_bias=False,
):
    # Sample a batch from memory
    if use_bias:
        states, next_states, actions, rewards, dones, biases = batch.to_torch_batch()
    else:
        states, next_states, actions, rewards, dones = batch.to_torch_batch()

    with torch.no_grad():
        if use_bias:
            next_state_action, next_state_log_pi, _ = agent.sample(
                next_states, bias=biases
            )
        else:
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

    if use_bias:
        pi, log_pi, _ = agent.sample(next_states, bias=biases)
    else:
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
        alpha_tlogs = agent.alpha.clone()  # For TensorboardX logs
    else:
        alpha_loss = torch.tensor(0.0).to(agent.device)
        alpha_tlogs = torch.tensor(agent.alpha)  # For TensorboardX logs

    soft_update(agent.critic_target, agent.critic, agent.tau)

    return (
        qf1_loss.item(),
        qf2_loss.item(),
        policy_loss.item(),
        alpha_loss.item(),
        alpha_tlogs.item(),
    )
