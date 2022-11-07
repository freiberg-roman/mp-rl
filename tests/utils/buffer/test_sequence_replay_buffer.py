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
        sim_qp_dim=4,
        sim_qv_dim=6,
        minimum_sequence_length=10,
        weight_mean_dim=3,
        weight_std_dim=3,
    )

    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
    for _ in range(10):
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
            (action, action),
            (action, action),
            mean,
            std,
        )

    for batch in buffer.get_iter(it=1, batch_size=4):
        assert batch.state.shape == (4, 5)

    assert len(buffer) == 10
    return True


def test_simple_trajectory():
    env = TestEnv(state_dim=8, action_dim=9)

    buffer = SequenceRB(
        capacity=10,
        state_dim=8,
        action_dim=9,
        sim_qp_dim=7,
        sim_qv_dim=9,
        minimum_sequence_length=4,
        weight_mean_dim=3,
        weight_std_dim=3,
    )

    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
    for _ in range(5):
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
            (action, action),
            (action, action),
            mean,
            std,
        )

    buffer.close_trajectory()

    for batch in buffer.get_true_k_sequence_iter(it=1, k=4, batch_size=5):
        assert batch.states.shape == (5, 4, 8)

    assert len(buffer) == 5
    assert buffer._valid_seq == [(0, 5, 0, 0, 2), (5, 5, 0, 2, 2)]
    assert np.all(buffer.get_valid_starts() == np.array([0, 1]))


def test_two_trajectories_one_overflow():
    env = TestEnv(state_dim=3, action_dim=19)

    buffer = SequenceRB(
        capacity=10,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
        minimum_sequence_length=3,
        weight_mean_dim=3,
        weight_std_dim=3,
    )

    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
    for _ in range(2):
        for _ in range(5):
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
                (action, action),
                (action, action),
                mean,
                std,
            )
        buffer.close_trajectory()
        buffer.close_trajectory()

    assert len(buffer) == 10
    assert buffer._valid_seq == [(0, 0, 1, 0, 0), (0, 5, 0, 0, 3), (5, 10, 0, 3, 6)]
    assert np.all(buffer.get_valid_starts() == np.array([0, 1, 2, 5, 6, 7]))

    buffer.add(
        state,
        next_state,
        action,
        1.0,
        False,
        sim_state,
        (action, action),
        (action, action),
        mean,
        std,
    )

    assert len(buffer) == 10
    assert buffer._valid_seq == [(0, 1, 1, 0, 0), (1, 5, 0, 1, 3), (5, 10, 0, 3, 6)]
    assert np.all(buffer.get_valid_starts() == np.array([1, 2, 5, 6, 7]))

    buffer.close_trajectory()

    assert len(buffer) == 10
    assert buffer._valid_seq == [
        (0, 1, 1, 0, 0),
        (1, 1, 1, 0, 0),
        (1, 5, 0, 1, 3),
        (5, 10, 0, 3, 6),
    ]
    assert np.all(buffer.get_valid_starts() == np.array([1, 2, 5, 6, 7]))


def test_overflow_buffer_multiple_times():
    env = TestEnv(state_dim=3, action_dim=19)
    buffer = SequenceRB(
        capacity=10,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
        minimum_sequence_length=3,
        weight_mean_dim=3,
        weight_std_dim=3,
    )

    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
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
            (action, action),
            (action, action),
            mean,
            std,
        )

    assert len(buffer) == 10
    assert np.all(buffer.get_valid_starts() == np.array([0, 1, 2, 3, 4, 5, 6, 7]))


def test_extinguish_other_sequence():
    env = TestEnv(state_dim=3, action_dim=19)
    buffer = SequenceRB(
        capacity=10,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
        minimum_sequence_length=4,
        weight_mean_dim=3,
        weight_std_dim=3,
    )

    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
    for i in [3, 3, 4]:
        for _ in range(i):
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
                (action, action),
                (action, action),
                mean,
                std,
            )
        buffer.close_trajectory()

    assert len(buffer) == 10
    assert np.all(buffer.get_valid_starts() == np.array([6]))

    for _ in range(6):
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
            (action, action),
            (action, action),
            mean,
            std,
        )
    buffer.close_trajectory()

    assert len(buffer) == 10
    assert np.all(buffer.get_valid_starts() == np.array([6, 0, 1, 2]))

    for _ in range(4):
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
            (action, action),
            (action, action),
            mean,
            std,
        )

    assert np.all(buffer.get_valid_starts() == np.array([0, 1, 2, 6]))


def test_store_load_srb():
    env = TestEnv(state_dim=3, action_dim=19)
    buffer = SequenceRB(
        capacity=10,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
        minimum_sequence_length=4,
        weight_mean_dim=3,
        weight_std_dim=3,
    )
    mean = np.random.standard_normal(size=(3))
    std = np.random.standard_normal(size=(3))
    for i in [3, 3, 4]:
        for _ in range(i):
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
                (action, action),
                (action, action),
                mean,
                std,
            )
        buffer.close_trajectory()

    buffer.store("./test_buffer/")
    buffer_two = SequenceRB(
        capacity=10,
        state_dim=3,
        action_dim=19,
        sim_qp_dim=2,
        sim_qv_dim=4,
        minimum_sequence_length=4,
        weight_mean_dim=3,
        weight_std_dim=3,
    )
    buffer_two.load("./test_buffer/")

    assert np.all(buffer.get_valid_starts() == buffer_two.get_valid_starts())
    assert np.all(buffer.states == buffer_two.states)
    assert np.all(buffer.next_states == buffer_two.next_states)
    assert np.all(buffer.actions == buffer_two.actions)
    assert np.all(buffer.rewards == buffer_two.rewards)
    assert np.all(buffer.dones == buffer_two.dones)
    assert np.all(buffer.sim_qps == buffer_two.sim_qps)
    assert np.all(buffer.sim_qvs == buffer_two.sim_qvs)
    assert np.all(buffer.des_qps == buffer_two.des_qps)
    assert np.all(buffer.des_qvs == buffer_two.des_qvs)
    assert np.all(buffer.des_qps_next == buffer_two.des_qps_next)
    assert np.all(buffer.des_qvs_next == buffer_two.des_qvs_next)
    assert np.all(buffer.weight_means == buffer_two.weight_means)
    assert np.all(buffer.weight_stds == buffer_two.weight_stds)

    import shutil

    shutil.rmtree("test_buffer/")
