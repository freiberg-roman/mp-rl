from typing import List, Optional, Tuple

import numpy as np

from mprl.utils.buffer.buffer_output import EnvStepsExtended
from mprl.utils.buffer.random_replay_buffer import RandomBatchIter
from mprl.utils.buffer.replay_buffer import EnvSteps, ReplayBuffer


class SequenceRB(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        sim_qpos_dim: int,
        sim_qvel_dim: int,
        minimum_sequence_length: int,
        des_qpos_dim: Optional[int] = None,
        weight_mean_dim: Optional[int] = None,
        weight_cov_dim: Optional[int] = None,
    ):
        self._s = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_s = np.empty((capacity, state_dim), dtype=np.float32)
        self._acts = np.empty((capacity, action_dim), dtype=np.float32)
        self._rews = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=bool)
        self._qposes = np.empty((capacity, sim_qpos_dim), dtype=np.float32)
        self._qvels = np.empty((capacity, sim_qvel_dim), dtype=np.float32)
        if des_qpos_dim is not None:
            self._des_qposes = np.empty((capacity, des_qpos_dim), dtype=np.float32)
        else:
            self._des_qposes = None
        if weight_mean_dim is not None:
            self._weight_means = np.empty((capacity, weight_mean_dim), dtype=np.float32)
        else:
            self._weight_means = None
        if weight_cov_dim is not None:
            self._weight_covs = np.empty((capacity, weight_cov_dim), dtype=np.float32)
        else:
            self._weight_covs = None

        self._capacity: int = 0
        self._max_capacity: int = capacity
        self._ind: int = 0
        self._current_seq = 0

        # sequence managing
        # (sequence_start, sequence_end, buffer_id, valid_starts_start, valid_starts_end)
        self._valid_seq: List = [(0, 0, 0, 0, 0)]
        self._min_seq_len = minimum_sequence_length
        self._valid_starts_buffers: Tuple[np.ndarray, np.ndarray] = [
            np.empty(capacity, dtype=np.int32),
            np.empty(capacity, dtype=np.int32),
        ]
        self._valid_starts_buffer_usages: List[Tuple[int, int], Tuple[int, int]] = [
            (0, 0),
            (0, 0),
        ]
        self._current_buffer = 0

    def add(
        self,
        state,
        next_state,
        action,
        reward,
        done,
        sim_state,
        des_q=None,
        weight_mean=None,
        weight_cov=None,
    ):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._qposes[self._ind, :] = sim_state[0]
        self._qvels[self._ind, :] = sim_state[1]
        if des_q is not None:
            self._des_qposes[self._ind, :] = des_q
        if weight_mean is not None:
            self._weight_means[self._ind, :] = weight_mean
        if weight_cov is not None:
            self._weight_covs[self._ind, :] = weight_cov
        self._capacity = min(self._capacity + 1, self._max_capacity)

        # adjust usage and sequence length
        s, e, buf_id, val_s, val_e = self._valid_seq[self._current_seq]
        current_valid_starts_len = self._compute_valid_starts(s, e + 1)
        if current_valid_starts_len > 0:
            assert self._current_buffer == buf_id
            assert val_e - val_s + 1 == current_valid_starts_len
            self._valid_starts_buffers[self._current_buffer][val_e] = (
                e - self._min_seq_len + 1
            )
            u_s, u_e = self._valid_starts_buffer_usages[self._current_buffer]
            self._valid_starts_buffer_usages[self._current_buffer] = (u_s, u_e + 1)
            self._valid_seq[self._current_seq] = (s, e + 1, buf_id, val_s, val_e + 1)
        else:
            assert val_e - val_s == 0
            self._valid_seq[self._current_seq] = (s, e + 1, buf_id, val_s, val_e)

        # check if sequence in front exists and adjust it
        self._adjust_next_sequence()

        # cyclic behavior for buffer
        if self._ind == self._max_capacity - 1:
            self._handle_buffer_overflow()
        else:
            self._ind += 1
        assert 0 <= self._ind < self._max_capacity

    def _handle_buffer_overflow(self):
        (buffer_usage_start, _) = self._valid_starts_buffer_usages[self._current_buffer]
        assert buffer_usage_start == 0
        (buffer_usage_start, buffer_usage_end) = self._valid_starts_buffer_usages[
            1 - self._current_buffer
        ]
        assert buffer_usage_start == buffer_usage_end
        _, e, _, _, _ = self._valid_seq[self._current_seq]
        assert e == self._max_capacity

        self._current_buffer = 1 - self._current_buffer
        self._ind = 0
        self._current_seq = 0
        self._valid_starts_buffer_usages[self._current_buffer] = (0, 0)
        self._valid_seq.insert(0, (0, 0, self._current_buffer, 0, 0))

    def _adjust_next_sequence(self):
        _, e, _, _, _ = self._valid_seq[self._current_seq]
        if len(self._valid_seq) != self._current_seq + 1:
            n_s, n_e, n_buf_id, n_val_s, n_val_e = self._valid_seq[
                self._current_seq + 1
            ]
            assert n_buf_id != self._current_buffer
            assert n_s + 1 == e
            if n_s + 1 == n_e:
                del self._valid_seq[self._current_seq + 1]
            else:
                if n_val_e - n_val_s > 0:
                    n_u_s, n_u_e = self._valid_starts_buffer_usages[n_buf_id]
                    assert n_u_e >= n_u_s
                    self._valid_starts_buffer_usages[n_buf_id] = (n_u_s + 1, n_u_e)
                    self._valid_seq[self._current_seq + 1] = (
                        n_s + 1,
                        n_e,
                        n_buf_id,
                        n_val_s + 1,
                        n_val_e,
                    )
                else:
                    self._valid_seq[self._current_seq + 1] = (
                        n_s + 1,
                        n_e,
                        n_buf_id,
                        n_val_s,
                        n_val_e,
                    )

    def close_trajectory(self):
        s, e, _, _, val_e = self._valid_seq[self._current_seq]
        if e - s == 0:
            return  # no empty trajectories
        self._valid_seq.insert(
            self._current_seq + 1,
            (self._ind, self._ind, self._current_buffer, val_e, val_e),
        )
        self._current_seq += 1

    def _compute_valid_starts(self, start, end):
        return max(0, end - start - self._min_seq_len + 1)

    def get_iter(self, it, batch_size):
        return RandomBatchIter(self, it, batch_size)

    def get_true_k_sequence_iter(self, it, k, batch_size):
        """
        Returns k-step sequences which are samples from steps from one sequence
        """
        return TrueKSequenceIter(self, it, k, batch_size=batch_size)

    def __getitem__(self, item):
        if self._des_qposes is not None:
            return EnvStepsExtended(
                self._s[item, :],
                self._next_s[item, :],
                self._acts[item, :],
                self._rews[item],
                self._dones[item],
                (self._qposes[item, :], self._qvels[item, :]),
                self._des_qposes[item, :],
                self._weight_means[item, :],
                self._weight_covs[item, :],
            )
        else:
            return EnvSteps(
                self._s[item, :],
                self._next_s[item, :],
                self._acts[item, :],
                self._rews[item],
                self._dones[item],
                (self._qposes[item, :], self._qvels[item, :]),
            )

    def __len__(self):
        return self._capacity

    @property
    def stored_sequences(self):
        return self._valid_seq

    @property
    def states(self):
        return self._s

    @property
    def next_states(self):
        return self._next_s

    @property
    def actions(self):
        return self._acts

    @property
    def rewards(self):
        return self._rews

    @property
    def dones(self):
        return self._dones

    @property
    def qposes(self):
        return self._qposes

    @property
    def qvels(self):
        return self._qvels

    @property
    def des_qposes(self):
        return self._des_qposes

    @property
    def weight_means(self):
        return self._weight_means

    @property
    def weight_covs(self):
        return self._weight_covs

    @property
    def capacity(self):
        return self._capacity

    def get_valid_starts(self):
        first_valid_starts = self._valid_starts_buffers[0][
            self._valid_starts_buffer_usages[0][0] : self._valid_starts_buffer_usages[
                0
            ][1]
        ]
        second_valid_starts = self._valid_starts_buffers[1][
            self._valid_starts_buffer_usages[1][0] : self._valid_starts_buffer_usages[
                1
            ][1]
        ]
        return np.concatenate((first_valid_starts, second_valid_starts))

    @property
    def is_extended(self):
        return self._des_qposes is not None


class TrueKSequenceIter:
    def __init__(self, buffer: SequenceRB, it: int, k: int, batch_size: int):
        self._buffer: SequenceRB = buffer
        self._it: int = it
        self._k: int = k
        self._current_it: int = 0
        self._batch_size: int = batch_size
        self._valid_starts = buffer.get_valid_starts()

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it and len(self._valid_starts) != 0:
            idxs = np.random.randint(0, len(self._valid_starts), self._batch_size)
            start_indices = self._valid_starts[idxs].repeat(self._k)
            increment_array = np.tile(np.arange(self._k), len(idxs))
            full_trajectory_indices = start_indices + increment_array
            full_trajectory_indices = full_trajectory_indices.reshape(
                (self._batch_size, self._k)
            )
            self._current_it += 1
            if self._buffer.is_extended:
                return EnvStepsExtended(
                    self._buffer.states[full_trajectory_indices, :],
                    self._buffer.next_states[full_trajectory_indices, :],
                    self._buffer.actions[full_trajectory_indices, :],
                    self._buffer.rewards[full_trajectory_indices],
                    self._buffer.dones[full_trajectory_indices],
                    (
                        self._buffer.qposes[full_trajectory_indices, :],
                        self._buffer.qvels[full_trajectory_indices, :],
                    ),
                    self._buffer.des_qposes[full_trajectory_indices, :],
                    self._buffer.weight_means[full_trajectory_indices, :],
                    self._buffer.weight_covs[full_trajectory_indices, :],
                )
            else:
                return EnvSteps(
                    self._buffer.states[full_trajectory_indices, :],
                    self._buffer.next_states[full_trajectory_indices, :],
                    self._buffer.actions[full_trajectory_indices, :],
                    self._buffer.rewards[full_trajectory_indices],
                    self._buffer.dones[full_trajectory_indices],
                    (
                        self._buffer.qposes[full_trajectory_indices, :],
                        self._buffer.qvels[full_trajectory_indices, :],
                    ),
                )
        else:
            raise StopIteration
