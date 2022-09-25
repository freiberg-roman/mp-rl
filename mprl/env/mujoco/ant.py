import numpy as np

from mprl.env.mujoco.mj_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntEnv(MujocoEnv):
    def __init__(
        self,
        base,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(self, base + xml_file, 5)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward
        costs = self.control_cost(action)

        self.current_steps += 1
        self._total_steps += 1
        observation = self._get_obs()
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
        reward = rewards - costs
        done = self.done
        timeout = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        return observation, reward, done, timeout, self.get_sim_state(), {}

    def random_action(self):
        return np.random.uniform(-1, 1, (8,))

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

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

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def steps_after_reset(self):
        return self.current_steps

    @property
    def get_jnt_names(self):
        return [
            "hip_1",
            "ankle_1",
            "hip_2",
            "ankle_2",
            "hip_3",
            "ankle_3",
            "hip_4",
            "ankle_4",
        ]

    @property
    def name(self) -> str:
        return "Ant"
