import numpy as np
from omegaconf import DictConfig

from ...env.mj_factory import create_mj_env
from .prediction import Prediction
from ...env.mujoco.mj_env import MujocoEnv
from ...utils.ds_helper import to_np


class GroundTruth(Prediction):

    def __init__(self, cfg_env: DictConfig):
        self.cfg_env = cfg_env
        self.env: MujocoEnv = create_mj_env(cfg_env)

    def next_state(self, states, actions):
        next_states = np.zeros_like(states)
        for i, sa in enumerate(zip(states, actions)):
            state, action = sa
            state = to_np(state)
            action = to_np(action)
            set_q, set_v = self.env.decompose(state, full_obs=True)
            self.env.set_state(set_q, set_v)
            next_state, _, _, _ = self.env.step(action)
            next_states[i] = next_state
        return next_states
