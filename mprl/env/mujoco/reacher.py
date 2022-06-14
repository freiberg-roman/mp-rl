import numpy as np

from mprl.env.mujoco.mj_env import MujocoEnv


class Reacher(MujocoEnv):
    def __init__(
        self,
        base,
        xml_file="reacher.xml",
    ):
        MujocoEnv.__init__(self, base + xml_file, 2)

    def step(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        self.current_steps += 1
        self._total_steps += 1
        done = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        return observation, reward, False, done

    def sample_random_action(self):
        return np.random.uniform(-1, 1, (2,))

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def get_forces(self):
        return super(Reacher, self).get_forces()[-6:]  # gravity

    @property
    def total_steps(self):
        return self._total_steps

    def decompose(self, state, full_obs=False):
        cos_pos = state[..., 0:2]
        sin_pos = state[..., 2:4]
        vel = state[..., 6:8]
        return np.arctan2(sin_pos, cos_pos), vel

    @property
    def steps_after_reset(self):
        return self.current_steps
