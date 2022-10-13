import warnings
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from mprl.utils.buffer.random_replay_buffer import RandomBatchIter
from mprl.utils.buffer.replay_buffer import EnvSteps, ReplayBuffer


class ValidStartsManager:
    def __init__(self, capacity: int, min_length_sequence: int):
        self.capacity = capacity
        self._buffers_usage = [(0, 0), (0, 0)]
        self._valid_starts_buffers: Tuple[np.ndarray, np.ndarray] = [
            np.empty(capacity, dtype=np.int32),
            np.empty(capacity, dtype=np.int32),
        ]
        self._assosiated_valid_seq_starts: List = [(0, 0, 0)]  # (start, end, buffer_id)
        self._current_seq = 0
        self._max_seq = 0
        self._buffer_used = 0
        self._min_len = min_length_sequence
        self._used_once = False

    def swap_buffers(self):
        if not self._used_once:
            return
        self._buffer_used = 1 - self._buffer_used
        s, e = self._buffers_usage[self._buffer_used]
        if s != e:
            self._buffer_used = 1 - self._buffer_used
            return
        self._buffers_usage[self._buffer_used] = (0, 0)
        self._current_seq = 0
        self.invalidate_seq(0)
        self._assosiated_valid_seq_starts.insert(0, (0, 0, self._buffer_used))

    def insert(self, seq: int):
        self._current_seq += 1
        assert seq == self._current_seq

        last_entry = self._buffers_usage[self._buffer_used][1]
        self._assosiated_valid_seq_starts.insert(
            self._current_seq, (last_entry, last_entry, self._buffer_used)
        )
        self._max_seq = max(self._current_seq, self._max_seq)

    def add_step(self, current_seq: int, current_start_end: Tuple[int, int]):
        self._used_once = True
        s, e = current_start_end
        assert self._current_seq == current_seq
        if e - s < self._min_len:
            return
        s_buf, e_buf, id_buf = self._assosiated_valid_seq_starts[self._current_seq]
        self._valid_starts_buffers[id_buf][e_buf] = e - self._min_len
        self._assosiated_valid_seq_starts[self._current_seq] = (
            s_buf,
            e_buf + 1,
            id_buf,
        )
        c_s, c_e = self._buffers_usage[id_buf]
        self._buffers_usage[id_buf] = (c_s, c_e + 1)

    def invalidate_seq(self, seq_id: int):
        if not self._used_once:
            return
        s, e, id = self._assosiated_valid_seq_starts[seq_id]
        assert id == 1 - self._buffer_used
        assert self._buffers_usage[id][0] == s
        end = self._buffers_usage[id][1]
        self._buffers_usage[id] = (e, end)
        del self._assosiated_valid_seq_starts[seq_id]

    def get_valid_starts(self):
        first_valid_starts = self._valid_starts_buffers[0][
            self._buffers_usage[0][0] : self._buffers_usage[0][1]
        ]
        second_valid_starts = self._valid_starts_buffers[1][
            self._buffers_usage[1][0] : self._buffers_usage[1][1]
        ]
        return np.concatenate((first_valid_starts, second_valid_starts))


class SequenceRB(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        sim_qpos_dim: int,
        sim_qvel_dim: int,
        min_length_sequence: int,
    ):
        self._s = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_s = np.empty((capacity, state_dim), dtype=np.float32)
        self._acts = np.empty((capacity, action_dim), dtype=np.float32)
        self._rews = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=bool)
        self._qposes = np.empty((capacity, sim_qpos_dim), dtype=np.float32)
        self._qvels = np.empty((capacity, sim_qvel_dim), dtype=np.float32)

        self._capacity: int = 0
        self._max_capacity: int = capacity
        self._ind: int = 0
        self._free_space_till: int = capacity
        self._current_seq = 0
        self._valid_seq: List = [(0, 0)]
        self.valid_seq_starts_manager: ValidStartsManager = ValidStartsManager(
            capacity, min_length_sequence
        )

    def add(self, state, next_state, action, reward, done, sim_state):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._qposes[self._ind, :] = sim_state[0]
        self._qvels[self._ind, :] = sim_state[1]
        self._capacity = min(self._capacity + 1, self._max_capacity)
        self._ind = (self._ind + 1) % self._max_capacity
        s, _ = self._valid_seq[self._current_seq]
        if self._ind == 0:
            self._valid_seq[self._current_seq] = (s, self._max_capacity)
            self.valid_seq_starts_manager.add_step(
                self._current_seq, (s, self._max_capacity)
            )
            self.close_trajectory()
        else:
            if self._free_space_till == self._ind:
                _, e = self._valid_seq[self._current_seq + 1]
                self.valid_seq_starts_manager.invalidate_seq(self._current_seq + 1)
                del self._valid_seq[self._current_seq + 1]
                self._free_space_till = e
            self._valid_seq[self._current_seq] = (s, self._ind)
            self.valid_seq_starts_manager.add_step(self._current_seq, (s, self._ind))

    def close_trajectory(self):
        if self._ind == 0:
            self._valid_seq[0] = (0, 0)
            self._current_seq = 0
            self._free_space_till = self._valid_seq[1][0]
            self.valid_seq_starts_manager.swap_buffers()
        else:
            self._valid_seq.insert(self._current_seq + 1, (self._ind, self._ind))
            self.valid_seq_starts_manager.insert(self._current_seq + 1)
            self._current_seq += 1

    def get_iter(self, it, batch_size):
        return RandomBatchIter(self, it, batch_size)

    def get_true_k_sequence_iter(self, it, k, batch_size):
        """
        Returns k-step sequences which are samples from steps from one sequence
        """
        return TrueKSequenceIter(self, it, k, batch_size=batch_size)

    def _remove_overlapping_seqs(self, seq_boundaries: Tuple[int, int]):
        start, end = seq_boundaries
        while end > self._free_space_till:
            s, e = self._valid_seq[self._last_seq_ind]
            self._free_space_till = e
            del self._valid_seq[self._last_seq_ind]

    def __getitem__(self, item):
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
    def capacity(self):
        return self._capacity

    def get_valid_starts(self):
        return self.valid_seq_starts_manager.get_valid_starts()


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
