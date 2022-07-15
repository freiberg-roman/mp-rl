import torch

from mprl.models.physics.prediction import Prediction
from mprl.utils import EnvSteps


def sac_policy_loss(agent: any, batch: EnvSteps):
    """
    Compute the policy loss for SAC.
    """
    states, actions, rewards, next_states, dones = batch.to_torch_batch()
    pi, log_pi, _, _ = agent.sample(states)
    qf1_pi, qf2_pi = agent.critic(states, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = (
        (agent.alpha * log_pi) - min_qf_pi
    ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    return policy_loss


class MixedMeanSACPolicyLoss:
    def __init__(self, model: Prediction):
        self.model = model

    def __call__(self, agent: any, batch: EnvSteps):
        states, actions, rewards, next_states, dones = batch.to_torch_batch()
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        agent.planner_train.re_init(
            weights[0], bc_pos=b_q, bc_vel=b_v, num_t=agent.num_steps + 1
        )
        next_states = states
        min_qf = 0
        called = 0
        for i, qv in enumerate(agent.planner_train):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_states)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(states, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            next_states = self.model.next_state(next_states, action)
            called = i
        min_qf /= called
        policy_loss = (-min_qf).mean()
        return policy_loss


class MixedWeightedSACPolicyLoss:
    def __init__(self, model: Prediction):
        self.model = model

    def __call__(self, agent: any, batch: EnvSteps):
        states, actions, rewards, next_states, dones = batch.to_torch_batch()
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        agent.planner_train.re_init(
            weights[0], bc_pos=b_q, bc_vel=b_v, num_t=agent.num_steps + 1
        )
        next_states = states
        min_qf = 0
        called = 0
        q_prob = torch.zeros(size=(*states.shape, agent.num_steps))
        for i, qv in enumerate(agent.planner_train):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_states)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(states, action)
            min_qf[..., i] = torch.min(qf1_pi, qf2_pi)
            next_states = self.model.next_state(next_states, action)
            _, log_prob, _, _ = agent.sample(next_states)
            q_prob[..., i] = log_prob.exp()
            called = i
        min_qf /= called
        # TODO: check if this is correct
        torch.nn.functional.normalize(
            q_prob, dim=-1
        )  # normalize q_prob over trajectories
        policy_loss = (-q_prob * min_qf).mean()
        return policy_loss
