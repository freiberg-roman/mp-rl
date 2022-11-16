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
        self._name = name
        self.info = None

    @property
    def get_jnt_names(self):
        return []

    def reset(self, time_out_after) -> np.ndarray:
        self.current_steps = 0
        self.time_out_after = time_out_after
        state = self.env.reset()
        return state, (None, None)

    def step(self, action: np.array):
        state, reward, failure, info = self.env.step(action)
        action_penalty = 0.05 * np.sum(action ** 2)
        reward -= action_penalty
        self.info = info
        self.current_steps += 1
        self._total_steps += 1
        timeout = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        return state, reward, failure, timeout

    def get_info(self):
        return self.info

    def get_sim_state(self):
        return (
            self.env.sim.data.qpos.ravel().copy(),
            self.env.sim.data.qvel.ravel().copy(),
        )

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def steps_after_reset(self) -> int:
        self._current_step

    @property
    def name(self) -> str:
        return self._name

    def reset_model(self):
        return self.env.reset_model()

    @property
    def dt(self):
        return self.env.model.opt.timestep * self.env.frame_skip

    @property
    def dof(self) -> int:
        return 4  # all are positional

    def random_action(self):
        return self.env.action_space.sample()

    def render(
        self,
        mode="human",
        width=480,
        height=480,
        camera_id=None,
        camera_name=None,
    ):
        return self.env.render(resolution=(width, height))

    def decompose_fn(
        self, states: np.ndarray, sim_states: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            states[..., :4],
            states[..., :4],
        )  # note: velocity is not given for xyz -> dummy used
