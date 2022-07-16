import numpy as np
from mprl.utils.buffer.sequence_replay_buffer import SequenceRB
from omegaconf import OmegaConf

config = OmegaConf.create(
    {
        "capacity": 25,
        "env": {
            "state_dim": 5,
            "action_dim": 3,
        },
    }
)


class TestEnv:
    """
    This is the interface that the replay buffer uses to interact with the environment.
    """

    def __init__(self, env_cfg):
        self.state_dim = env_cfg.state_dim
        self.action_dim = env_cfg.action_dim

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


def test_simple_random_buffer():
    """
    Test that the random buffer works as expected.
    """
    env = TestEnv(config.env)
    buffer = SequenceRB(config)

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

    for batch in buffer.get_iter(it=1, batch_size=4):
        assert batch.states.shape == (4, 5)

    assert len(buffer) == 10


def test_trajectory_buffer():
    """
    Test that the trajectory buffer works as expected.
    """
    env = TestEnv(config.env)
    buffer = SequenceRB(config)

    for _ in range(4):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

    buffer.close_trajectory()

    for _ in range(5):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

    buffer.close_trajectory()

    for _ in range(6):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

    buffer.close_trajectory()

    assert len(buffer) == 15
    assert buffer.stored_sequences == [(0, 4), (4, 9), (9, 15), (15, 15)]

    for batch in buffer.get_true_k_sequence_iter(it=1, k=3, batch_size=4):
        assert batch.states.shape == (4, 3, 5)

    for batch in buffer.get_true_k_sequence_iter(it=2, k=6, batch_size=4):
        assert (
            batch.states[0, 0, 0] == batch.states[1, 0, 0]
        )  # there is only one sequence of this size -> all are the same

    for _ in buffer.get_true_k_sequence_iter(it=2, k=7, batch_size=4):
        raise AssertionError()  # sequence of this size should not be possible


def test_traj_overflow():
    """
    Test that the trajectory buffer can handle trajectories that are too long.
    We expect cyclic behavior i.e. the last trajectory is split into two, where
    the later part is appended to the beginning of the buffer.
    """
    env = TestEnv(config.env)
    buffer = SequenceRB(config)

    for _ in range(10):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

    buffer.close_trajectory()

    for _ in range(20):
        state = env.rnd_state()
        action = env.rnd_action()
        next_state = env.rnd_state()

        buffer.add(state, next_state, action, 1.0, False)

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

        buffer.add(state, next_state, action, 1.0, False)

    buffer.close_trajectory()
    assert buffer.stored_sequences == [
        (0, 5),
        (5, 15),
        (15, 15),
    ]  # trajectories are split
