import numpy as np
from omegaconf import DictConfig

from ...env.mj_factory import create_mj_env
from ...env.mujoco.mj_env import MujocoEnv
from ...utils.ds_helper import to_np
from .prediction import Prediction


class GroundTruth(Prediction):
    def __init__(self, cfg_env: DictConfig):
        self.cfg_env = cfg_env
        self.env: MujocoEnv = create_mj_env(cfg_env)
        self.state_dim = cfg_env.state_dim
        self.env.reset()

    def next_state(self, sim_states, actions):
        qposes = to_np(sim_states[0])
        qvels = to_np(sim_states[1])
        actions = to_np(actions)
        next_states = np.empty((len(actions), self.state_dim))
        for i, pva in enumerate(zip(qposes, qvels, actions)):
            p, v, a = pva
            self.env.set_state(p, v)
            next_state, _, _, _, _ = self.env.step(a)
            next_states[i] = next_state
        return next_states

    def update_parameters(self, batch):
        pass
