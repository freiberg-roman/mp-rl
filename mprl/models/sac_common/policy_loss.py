from copy import deepcopy

import torch
import torch.nn.functional as F

from mprl.models.physics.prediction import Prediction
from mprl.models.sac_mixed import SACMixed
from mprl.utils import EnvSteps
from mprl.utils.ds_helper import to_ts


def sac_policy_loss(agent: any, batch: EnvSteps):
    """
    Compute the policy loss for SAC.
    """
    states, next_states, actions, rewards, dones = batch.to_torch_batch()
    pi, log_pi, _, _ = agent.sample(states)
    qf1_pi, qf2_pi = agent.critic(states, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = (
        (agent.alpha * log_pi) - min_qf_pi
    ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    return policy_loss


class MixedMeanSACModelPolicyLoss:
    def __init__(self, model: Prediction):
        self.model = model

    def __call__(self, agent: any, batch: EnvSteps):
        # dimension (batch_size, data_dimension)
        states, next_states, actions, rewards, dones = batch.to_torch_batch()
        # dimension (1, batch_size, weight_dimension)
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        agent.planner_train.re_init(
            weights[0], bc_pos=b_q, bc_vel=b_v, num_t=agent.num_steps
        )
        next_states = states
        min_qf = 0
        for _, qv in enumerate(agent.planner_train):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_states)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val: dimension (batch_size, 1)
            qf1_pi, qf2_pi = agent.critic(next_states, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            next_states = to_ts(self.model.next_state(next_states, action))
        min_qf /= agent.num_steps
        policy_loss = (-min_qf).mean()
        return policy_loss


class MixedMeanSACOffPolicyLoss:
    def __init__(self):
        pass

    def __call__(self, agent: any, batch: EnvSteps):
        # dimensions (batch_size, sequence_len, data_dimension)
        states, next_states, actions, rewards, dones = batch.to_torch_batch()
        # dimension (1, batch_size, sequence_len, weight_dimension)
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        agent.planner_train.re_init(
            weights[0, :, 0, :],
            bc_pos=b_q[:, 0, :],
            bc_vel=b_v[:, 0, :],
            num_t=agent.num_steps,
        )
        next_s = states[:, 0, :]
        min_qf = 0
        for i, qv in enumerate(agent.planner_train):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_s)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(next_s, action)
            min_qf += torch.min(qf1_pi, qf2_pi)
            # we simply take the sequence as fixed
            next_s = next_states[:, i, :]
        min_qf /= agent.num_steps
        policy_loss = (-min_qf).mean()
        return policy_loss


class MixedWeightedSACModelPolicyLoss:
    def __init__(self, model: Prediction):
        self.model = model

    def __call__(self, agent: SACMixed, batch: EnvSteps):
        # dimension (batch_size, data_dimension)
        states, next_states, actions, rewards, dones = batch.to_torch_batch()
        # dimension (1, batch_size, weight_dimension)
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        planner_local = deepcopy(agent.planner_train)
        planner_local.re_init(weights[0], bc_pos=b_q, bc_vel=b_v, num_t=agent.num_steps)
        next_states = states
        # dimensions (batch_size, sequence_len, 1)
        q_prob = torch.zeros(size=(len(states), agent.num_steps, 1))
        min_qf = torch.zeros_like(q_prob)
        for i, qv in enumerate(planner_local):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_states)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(next_states, action)
            min_qf[:, i, :] = torch.min(qf1_pi, qf2_pi)
            next_states = to_ts(self.model.next_state(next_states, action))
            # dimension (batch_size, 1)
            q_prob[:, i, :] = agent.prob(next_states, weights[0])
        min_qf /= agent.num_steps
        q_prob = F.normalize(q_prob, p=1.0, dim=1)
        policy_loss = (-q_prob * min_qf).mean()
        return policy_loss


class MixedWeightedSACOffPolicyLoss:
    def __init__(self):
        pass

    def __call__(self, agent: SACMixed, batch: EnvSteps):
        # dimension (batch_size, sequence_len, data_dimension)
        states, next_states, actions, rewards, dones = batch.to_torch_batch()
        # dimension (1, batch_size, sequence_len, weight_dimension)
        weights, _ = agent.select_weights_and_time(states)
        b_q, b_v = agent.decompose_fn(states)
        planner_local = deepcopy(agent.planner_train)
        planner_local.re_init(
            weights[0, :, 0, :],
            bc_pos=b_q[:, 0, :],
            bc_vel=b_v[:, 0, :],
            num_t=agent.num_steps,
        )
        next_s = states[:, 0, :]
        # dimensions (batch_size, sequence_len, 1)
        q_prob = torch.zeros(size=(len(states), agent.num_steps, 1))
        min_qf = torch.zeros_like(q_prob)
        for i, qv in enumerate(planner_local):
            q, v = qv
            b_q, b_v = agent.decompose_fn(next_s)
            action, _ = agent.ctrl.get_action(q, v, b_q, b_v)

            # compute q val
            qf1_pi, qf2_pi = agent.critic(next_s, action)
            min_qf[:, i, :] = torch.min(qf1_pi, qf2_pi)
            next_s = next_states[:, i, :]
            # dimension (batch_size, 1)
            q_prob[:, i, :] = agent.prob(next_s, weights[0])
        min_qf /= agent.num_steps
        q_prob = F.normalize(q_prob, p=1.0, dim=1)
        policy_loss = (-q_prob * min_qf).mean()
        return policy_loss
