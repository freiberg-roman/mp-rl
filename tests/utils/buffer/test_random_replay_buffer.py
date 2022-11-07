import numpy as np

from mprl.utils import RandomRB


class TestEnv:
    """
    This is the interface that the replay buffer uses to interact with the environment.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def rnd_action(self):
        """
        Returns a random action.
        """
        return np.random.standard_normal(size=(self.action_dim))

    def rnd_state(self):
        """
        Returns a random state.
        """
        return np.random.standard_normal(size=(self.state_dim))

    def rnd_sim_state(self):
        """
        Returns a random sim state.
        """
        return (
            np.random.standard_normal(size=(self.state_dim - 1)),
            np.random.standard_normal(size=(self.state_dim + 1)),
        )


def test_load_store_rrb():
    env = TestEnv(state_dim=3, action_dim=19)
    buffer = RandomRB(
        capacity=13,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
    )

    for _ in range(1000):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()
        buffer.add(
            state,
            next_state,
            action,
            1.0,
            False,
            sim_state,
        )

    buffer.store(".test_rrb/")
    buffer_two = RandomRB(
        capacity=13,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
    )
    buffer_two.load(".test_rrb/")

    assert len(buffer) == len(buffer_two)
    assert np.all(buffer._s == buffer_two._s)
    assert np.all(buffer._next_s == buffer_two._next_s)
    assert np.all(buffer._acts == buffer_two._acts)
    assert np.all(buffer._rews == buffer_two._rews)
    assert np.all(buffer._dones == buffer_two._dones)
    assert np.all(buffer._sim_qps == buffer_two._sim_qps)
    assert np.all(buffer._sim_qvs == buffer_two._sim_qvs)
    assert buffer._capacity == buffer_two._capacity
    assert buffer._max_capacity == buffer_two._max_capacity
    assert buffer._ind == buffer_two._ind

    import shutil

    shutil.rmtree(".test_rrb/")
