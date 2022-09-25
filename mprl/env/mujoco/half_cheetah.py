from typing import Tuple

import numpy as np

from mprl.env.mujoco.mj_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(MujocoEnv):
    def __init__(
        self,
        base,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        MujocoEnv.__init__(self, base + xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        observation = self._get_obs()
        reward = forward_reward - ctrl_cost

        self.current_steps += 1
        self._total_steps += 1
        done = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        return observation, reward, False, done, self.get_sim_state(), {}

    def random_action(self):
        return np.random.uniform(-1, 1, (6,))

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + np.random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * np.random.standard_normal(
            self.model.nv
        )

        self.set_sim_state((qpos, qvel))

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def get_forces(self):
        return super(HalfCheetahEnv, self).get_forces()[-6:]  # gravity

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def steps_after_reset(self):
        return self.current_steps

    @property
    def get_jnt_names(self):
        return ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    def decompose_fn(
        self, states: np.ndarray, sim_states: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        qpos_joint = sim_states[0][..., self.qpos_idx]
        qvel_joint = sim_states[1][..., self.qvel_idx]
        return qpos_joint, qvel_joint
