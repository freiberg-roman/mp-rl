import numpy as np

from mprl.env.mujoco.mj_env import MujocoEnv


class ReacherEnv(MujocoEnv):
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
        return observation, reward, False, done, self.get_sim_state(), {}

    def random_action(self):
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
            np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        )
        # We just want targets in the upper half of the board
        while True:
            self.goal = np.zeros((2,))
            self.goal[0] = np.random.uniform(low=-0.2, high=0.2, size=(1,))
            self.goal[1] = np.random.uniform(low=0.0, high=0.2, size=(1,))
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + np.random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_sim_state((qpos, qvel))
        return self._get_obs()

    def set_robot_to(self, qpos):
        qpos_state = np.zeros((4,))
        qpos_state[:2] = qpos
        self.set_state(qpos_state, qvel=np.zeros((4,)))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def steps_after_reset(self):
        return self.current_steps

    @property
    def get_jnt_names(self):
        return ["joint0", "joint1"]

    @property
    def name(self) -> str:
        return "Reacher"
