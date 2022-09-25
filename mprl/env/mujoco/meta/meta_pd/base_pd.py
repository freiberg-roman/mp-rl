from abc import ABC
from typing import Tuple

import numpy as np

from ..base import SawyerXYZEnv


class SawyerPD(SawyerXYZEnv, ABC):
    def __init__(self, path, hand_low, hand_high):
        super().__init__(path, hand_low=hand_low, hand_high=hand_high)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.curr_path_length += 1
        self.current_steps += 1
        self._total_steps += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        done = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        self._last_stable_obs = self._get_obs()
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        return self._last_stable_obs, reward, False, done, self.get_sim_state(), info

    def random_action(self):
        return np.random.uniform(-1, 1, (9,))

    def _reset_hand(self, steps=50):
        for _ in range(steps):
            self.data.mocap_pos[0, :] = self.hand_init_pos
            self.data.mocap_quat[0, :] = np.array([[1, 0, 1, 0]])
            self.do_simulation([0.0] * 7 + [-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    @property
    def dof(self) -> int:
        return 9

    @property
    def get_jnt_names(self):
        return [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
            "r_close",
            "l_close",
        ]

    def decompose_fn(
        self, states: np.ndarray, sim_states: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        qpos_joint = sim_states[0][..., self.qpos_idx]
        qvel_joint = sim_states[1][..., self.qvel_idx]
        return qpos_joint, qvel_joint

    def evaluate_state(self, obs, action):
        raise NotImplementedError
