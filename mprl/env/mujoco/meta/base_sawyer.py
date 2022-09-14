import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mprl.env.mujoco.mj_env import MujocoEnv


class BaseSawyer(MujocoEnv):
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        self.current_steps += 1
        self._total_steps += 1
        done = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        obs = self._get_obs()
        reward = self.evaluate_state(obs, action)
        return obs, reward, False, done, self.get_sim_state()

    def _reset_hand(self, steps=50):
        for _ in range(steps):
            self.data.mocap_pos[0, :] = self.hand_init_pos
            self.data.mocap_quat[0, :] = np.array([[1, 0, 1, 0]])
            self.do_simulation([0] * 7 + [-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def sample_random_action(self):
        return np.random.uniform(-1, 1, (9,))

    def evaluate_state(self, obs, action):
        reward, reach_dist, in_place = self.compute_reward(obs, action)
        return reward

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def steps_after_reset(self):
        return self.current_steps

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

    def _get_quat_objects(self):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        mat = self.data.geom_xmat[id].reshape((3, 3))
        return Rotation.from_matrix(mat).as_quat()