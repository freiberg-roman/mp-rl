import random
from typing import Tuple

import metaworld
import numpy as np
from mujoco_py import MjSimState

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
        """Meta world sim state

        We will encode _site_targets in qpos and mocap information in qvel
        Layout qpos[0:16] sim state qpos, qpos[16:] target site pos
        qvel[0:15] sim state qvel,
        :return:
        """
        state = self.env.get_env_state()
        qvel = state[0].qvel
        mocap_pos = state[1][0][0]
        mocap_quat = state[1][1][0]
        qvel_all = np.concatenate((qvel, mocap_pos, mocap_quat))

        qpos = state[0].qpos
        if self._name == "reach-v2":
            target = self.env._target_pos
            qpos_all = np.concatenate((qpos, target))
        elif self._name == "window-open-v2":
            ...
        elif self.name == "button-press-v2":
            ...
        else:
            raise ValueError("No such environment!")

        return qpos_all, qvel_all

    def set_sim_state(self, sim_state: Tuple[np.ndarray, np.ndarray]):
        qpos_all, qvel_all = sim_state
        mj_sim_state = MjSimState(0.0, qpos_all[0:16], qvel_all[0:15], None, {})
        self.env.sim.set_state(mj_sim_state)

        mocap_pos, mocap_quat = (qvel_all[15:18])[None], (qvel_all[18:22])[None]
        self.env.data.set_mocap_pos("mocap", mocap_pos)
        self.env.data.set_mocap_quat("mocap", mocap_quat)
        self.env.sim.forward()

        if self._name == "reach-v2":
            self.env._target_pos = qpos_all[16:19]
        elif self._name == "window-open-v2":
            ...
        elif self.name == "button-press-v2":
            ...
        else:
            raise ValueError("No such environment!")

    def reset_model(self):
        return self.env.reset_model()

    @property
    def dt(self):
        return self.env.model.opt.timestep * self.env.frame_skip

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
