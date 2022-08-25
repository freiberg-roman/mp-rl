import warnings
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from mprl.utils.buffer.random_replay_buffer import RandomBatchIter
from mprl.utils.buffer.replay_buffer import EnvSteps, ReplayBuffer


class SequenceRB(ReplayBuffer):
    def __init__(self, cfg: DictConfig):
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)
        self._qposes = np.empty((cfg.capacity, cfg.env.sim_qpos_dim), dtype=np.float32)
        self._qvels = np.empty((cfg.capacity, cfg.env.sim_qvel_dim), dtype=np.float32)

        self._capacity: int = 0
        self._max_capacity: int = cfg.capacity
        self._ind: int = 0
        self._free_space_till: int = cfg.capacity
        self._current_seq = 0
        self._valid_seq: List = [(0, 0)]

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
            self.close_trajectory()
        else:
            if self._free_space_till == self._ind:
                _, e = self._valid_seq[self._current_seq + 1]
                del self._valid_seq[self._current_seq + 1]
                self._free_space_till = e
            self._valid_seq[self._current_seq] = (s, self._ind)

    def close_trajectory(self):
        if self._ind == 0:
            self._valid_seq[0] = (0, 0)
            self._current_seq = 0
            self._free_space_till = self._valid_seq[1][0]
        else:
            self._valid_seq.insert(self._current_seq + 1, (self._ind, self._ind))
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


class TrueKSequenceIter:
    def __init__(self, buffer: SequenceRB, it: int, k: int, batch_size: int):
        self._buffer: SequenceRB = buffer
        self._it: int = it
        self._k: int = k
        self._current_it: int = 0
        self._batch_size: int = batch_size
        self._valid_starts: List = []
        for (start, end) in self._buffer.stored_sequences:
            if end - start < self._k:
                continue
            self._valid_starts.extend(list(range(start, end - self._k + 1)))
        self._valid_starts = np.array(self._valid_starts)

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
