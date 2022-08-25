import torch
import torch.nn.functional as F

from mprl.utils import EnvSteps


def sac_critic_loss(agent: any, batch: EnvSteps):
    """
    Compute the critic loss for SAC.
    """
    states, next_states, actions, rewards, dones, _ = batch.to_torch_batch()
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
    return qf_loss


def sac_mp_critic_loss(agent: any, batch: EnvSteps):
    """
    Compute the critic loss for SAC.
    """
    states, next_states, actions, rewards, dones, sim_states = batch.to_torch_batch()
    with torch.no_grad():
        next_state_action, next_state_log_pi, _, _ = agent.sample(
            next_states, sim_states
        )
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
    return qf_loss


def sac_critic_loss_sequenced(agent: any, batch: EnvSteps):
    """
    Compute the critic loss for SAC.
    """
    states, next_states, actions, rewards, dones = batch.to_torch_batch()
    states, next_states, actions, rewards, dones = (
        states[:, 0, :],
        next_states[:, 0, :],
        actions[:, 0, :],
        rewards[:, 0, :],
        dones[:, 0, :],
    )
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
    return qf_loss
