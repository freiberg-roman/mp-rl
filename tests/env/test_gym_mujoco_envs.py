import numpy as np
from mprl.env import MujocoFactory, EnvConfigGateway

envs = [
    "HalfCheetah",
    "Ant",
    "Hopper",
    "Reacher",
]


class EnvConfigGatewayMock(EnvConfigGateway):
    def __init__(self, env_name):
        self.env_name = env_name

    def get_env_name(self) -> str:
        return self.env_name


def test_basic_routine():
    for env_name in envs:
        cfg = EnvConfigGatewayMock(env_name)
        factory = MujocoFactory(cfg)
        env = factory.create()

        env.reset(time_out_after=100)
        for _ in range(10):
            env.step(env.random_action())

        assert env.total_steps == 10
        env.full_reset()
        assert env.total_steps == 0


def test_reset_sim_states():
    for env_name in envs:
        cfg = EnvConfigGatewayMock(env_name)
        factory = MujocoFactory(cfg)
        env = factory.create()

        state, sim_state = env.reset(time_out_after=100)
        action = env.random_action()
        new_state, _, _, _ = env.step(action)
        for _ in range(10):
            env.step(action)

        env.set_sim_state(sim_state)
        state_after_sss, _, _, _, _, _ = env.step(action)
        assert np.allclose(state_after_sss, new_state)


def test_decompose_fn():
    for env_name in envs:
        cfg = EnvConfigGatewayMock(env_name)
        factory = MujocoFactory(cfg)
        env = factory.create()
        state, sim_state = env.reset(time_out_after=100)

        qpos, qvel = env.decompose_fn(state, sim_state)
        assert env.dof == len(qpos) == len(qvel)
