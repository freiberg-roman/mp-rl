import random
from typing import Tuple

import metaworld
import numpy as np

from mprl.env.mujoco.mj_env import MujocoEnv


class OriginalMetaWorld(MujocoEnv):
    def __init__(self, name):
        self.ml1 = metaworld.ML1(name)
        self.env = self.ml1.train_classes[name]()
        self.task = random.choice(self.ml1.train_tasks)
        self.env.set_task(self.task)
        self._total_steps = 0
        self.current_steps = 0

    @property
    def get_jnt_names(self):
        return []

    def reset(self, time_out_after) -> np.ndarray:
        self.current_steps = 0
        self.time_out_after = time_out_after
        state = self.env.reset()
        return state, (None, None)

    def step(self, action: np.array):
        state, reward, done, info = self.env.step(action)
        self.current_steps += 1
        self._total_steps += 1
        timeout = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        return state, reward, done, timeout, self.get_sim_state(), info

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def steps_after_reset(self) -> int:
        self._current_step

    @property
    def name(self) -> str:
        return "OriginalMetaReacher"

    def get_sim_state(self):
        return np.zeros(16), np.zeros(15)

    def set_sim_state(self, sim_state: Tuple[np.ndarray, np.ndarray]):
        self.env.set_state(*sim_state)

    def reset_model(self):
        return self.env.reset_model()

    def dt(self):
        return self.env.model.opt.timestep * 5

    def random_action(self):
        return self.env.action_space.sample()
