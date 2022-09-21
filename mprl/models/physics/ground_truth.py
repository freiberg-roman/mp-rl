from typing import Tuple

import numpy as np

from mprl.env.mp_rl_environment import MPRLEnvironment
from mprl.utils.ds_helper import to_np

from ..common.interfaces import Predictable


class GroundTruthPrediction(Predictable):
    def __init__(self, env: MPRLEnvironment):
        self.env: MPRLEnvironment = env
        self.env.full_reset()

    def next_state(
        self,
        states: np.ndarray,
        sim_states: Tuple[np.ndarray, np.ndarray],
        actions: np.ndarray,
    ):
        """
        :param states: (batch_size, state_dim)
        :param sim_states: Tuple of (batch_size, sim_qpos_dim), (batch_size, sim_qvel_dim)
        :param actions: (batch_size, action_dim)
        :return:
        """
        qposes = to_np(sim_states[0])
        qvels = to_np(sim_states[1])
        actions = to_np(actions)
        next_states = np.empty_like(states)
        next_qposes = np.empty_like(qposes)
        next_qvels = np.empty_like(qvels)
        for i, qva in enumerate(zip(qposes, qvels, actions)):
            q, v, a = qva
            self.env.set_sim_state((q, v))
            next_state, _, _, _, next_sim_states = self.env.step(a)
            next_states[i] = next_state
            next_qposes[i] = next_sim_states[0]
            next_qvels[i] = next_sim_states[1]
        return next_states, (next_qposes, next_qvels)
