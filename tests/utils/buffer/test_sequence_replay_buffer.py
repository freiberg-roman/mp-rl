import numpy as np
from mprl.utils.buffer.sequence_replay_buffer import SequenceRB


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


def test_simple_random_buffer():
    """
    Test that the random buffer works as expected.
    """
    env = TestEnv(state_dim=5, action_dim=2)
    buffer = SequenceRB(
        capacity=20,
        state_dim=5,
        action_dim=2,
        sim_qpos_dim=4,
        sim_qvel_dim=6,
        min_length_sequence=10,
    )

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    for batch in buffer.get_iter(it=1, batch_size=4):
        assert batch.states.shape == (4, 5)

    assert len(buffer) == 10


def test_trajectory_buffer():
    """
    Test that the trajectory buffer works as expected.
    """
    env = TestEnv(state_dim=6, action_dim=3)
    buffer = SequenceRB(
        capacity=20,
        state_dim=6,
        action_dim=3,
        sim_qpos_dim=5,
        sim_qvel_dim=7,
        min_length_sequence=3,
    )

    for _ in range(4):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()

    for _ in range(5):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()

    for _ in range(6):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()

    assert len(buffer) == 15
    assert buffer.stored_sequences == [(0, 4), (4, 9), (9, 15), (15, 15)]
    assert np.all(buffer.get_valid_starts() == np.array([0, 1, 4, 5, 6, 9, 10, 11, 12]))

    for batch in buffer.get_true_k_sequence_iter(it=1, k=3, batch_size=4):
        assert batch.states.shape == (4, 3, 6)


def test_traj_overflow():
    """
    Test that the trajectory buffer can handle trajectories that are too long.
    We expect cyclic behavior i.e. the last trajectory is split into two, where
    the later part is appended to the beginning of the buffer.
    """
    env = TestEnv(state_dim=6, action_dim=3)
    buffer = SequenceRB(
        capacity=25,
        state_dim=6,
        action_dim=3,
        sim_qpos_dim=5,
        sim_qvel_dim=7,
        min_length_sequence=3,
    )

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()

    for _ in range(20):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()
    # since the remaining stored pairs are still valid the whole buffer should still be used.
    assert len(buffer) == 25  # overflow
    assert buffer.stored_sequences == [
        (0, 5),
        (5, 5),
        (10, 25),
    ]  # trajectories are split

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()
    assert buffer.stored_sequences == [
        (0, 5),
        (5, 15),
        (15, 15),
    ]  # trajectories are split
    assert np.all(
        buffer.get_valid_starts() == np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12])
    )


def test_close_trajectory_immediately():
    env = TestEnv(state_dim=6, action_dim=3)
    buffer = SequenceRB(
        capacity=25,
        state_dim=6,
        action_dim=3,
        sim_qpos_dim=5,
        sim_qvel_dim=7,
        min_length_sequence=3,
    )

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)

    buffer.close_trajectory()
    buffer.close_trajectory()

    for _ in range(5):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()
        sim_state = env.rnd_sim_state()

        buffer.add(state, next_state, action, 1.0, False, sim_state)
    buffer.close_trajectory()
    assert buffer.stored_sequences == [
        (0, 10),
        (10, 10),
        (10, 15),
        (15, 15),
    ]  # trajectories are split
    assert np.all(
        buffer.get_valid_starts() == np.array([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12])
    )
