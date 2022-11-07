from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mprl.utils.buffer.random_replay_buffer import RandomBatchIter

from ..ds_helper import to_np
from ..serializable import Serializable
from .buffer_output import EnvSequence


class SequenceRB(Serializable):
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        sim_qp_dim: int,
        sim_qv_dim: int,
        minimum_sequence_length: int,
        weight_mean_dim: int,
        weight_std_dim: int,
    ):
        self._s = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_s = np.empty((capacity, state_dim), dtype=np.float32)
        self._acts = np.empty((capacity, action_dim), dtype=np.float32)
        self._rews = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=bool)
        self._sim_qps = np.empty((capacity, sim_qp_dim), dtype=np.float32)
        self._sim_qvs = np.empty((capacity, sim_qv_dim), dtype=np.float32)
        self._des_qps = np.empty((capacity, action_dim), dtype=np.float32)
        self._des_qvs = np.empty((capacity, action_dim), dtype=np.float32)
        self._des_qps_next = np.empty((capacity, action_dim), dtype=np.float32)
        self._des_qvs_next = np.empty((capacity, action_dim), dtype=np.float32)
        self._weight_means = np.empty((capacity, weight_mean_dim), dtype=np.float32)
        self._weight_stds = np.empty((capacity, weight_std_dim), dtype=np.float32)

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

    def store_under(self):
        return "srb"

    def store(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(path + "state.npy", self._s)
        np.save(path + "next_states.npy", self._next_s)
        np.save(path + "actions.npy", self._acts)
        np.save(path + "actions.npy", self._acts)
        np.save(path + "rewards.npy", self._rews)
        np.save(path + "dones.npy", self._dones)
        np.save(path + "sim_qps.npy", self._sim_qps)
        np.save(path + "sim_qvs.npy", self._sim_qvs)
        np.save(path + "des_qps.npy", self._des_qps)
        np.save(path + "des_qvs.npy", self._des_qvs)
        np.save(path + "des_qps_next.npy", self._des_qps_next)
        np.save(path + "des_qvs_next.npy", self._des_qvs_next)
        np.save(path + "weight_means.npy", self._weight_means)
        np.save(path + "weight_stds.npy", self._weight_stds)
        np.save(path + "capacity.npy", np.array([self._capacity], dtype=int))
        np.save(path + "max_capacity.npy", np.array([self._max_capacity], dtype=int))
        np.save(path + "index.npy", np.array([self._ind], dtype=int))
        np.save(path + "current_sequence.npy", np.array([self._current_seq], dtype=int))
        np.save(path + "valid_sequence.npy", np.array([self._valid_seq], dtype=int))
        np.save(
            path + "min_sequence_length.npy", np.array([self._min_seq_len], dtype=int)
        )
        np.save(path + "buffer_one.npy", self._valid_starts_buffers[0])
        np.save(path + "buffer_two.npy", self._valid_starts_buffers[1])
        np.save(
            path + "usage_one_one.npy",
            np.array([self._valid_starts_buffer_usages[0][0]], dtype=int),
        )
        np.save(
            path + "usage_one_two.npy",
            np.array([self._valid_starts_buffer_usages[0][1]], dtype=int),
        )
        np.save(
            path + "usage_two_one.npy",
            np.array([self._valid_starts_buffer_usages[1][0]], dtype=int),
        )
        np.save(
            path + "usage_two_two.npy",
            np.array([self._valid_starts_buffer_usages[1][1]], dtype=int),
        )
        np.save(
            path + "current_buffer.npy", np.array([self._current_buffer], dtype=int)
        )

    # Load model parameters
    def load(self, path: str) -> None:
        self._s = np.load(path + "state.npy")
        self._next_s = np.load(path + "next_states.npy")
        self._acts = np.load(path + "actions.npy")
        self._rews = np.load(path + "rewards.npy")
        self._dones = np.load(path + "dones.npy")
        self._sim_qps = np.load(path + "sim_qps.npy")
        self._sim_qvs = np.load(path + "sim_qvs.npy")
        self._des_qps = np.load(path + "des_qps.npy")
        self._des_qvs = np.load(path + "des_qvs.npy")
        self._des_qps_next = np.load(path + "des_qps_next.npy")
        self._des_qvs_next = np.load(path + "des_qvs_next.npy")
        self._weight_means = np.load(path + "weight_means.npy")
        self._weight_stds = np.load(path + "weight_stds.npy")
        self._capacity = np.load(path + "capacity.npy").item()
        self._max_capacity = np.load(path + "max_capacity.npy").item()
        self._ind = np.load(path + "index.npy").item()
        self._current_seq = np.load(path + "current_sequence.npy")
        valid_seq_array = np.load(path + "valid_sequence.npy")[0]
        self._valid_seq = []
        for i in range(len(valid_seq_array)):
            self._valid_seq.append(
                (
                    valid_seq_array[i][0],
                    valid_seq_array[i][1],
                    valid_seq_array[i][2],
                    valid_seq_array[i][3],
                    valid_seq_array[i][4],
                )
            )

        self._min_seq_len = np.load(path + "min_sequence_length.npy").item()
        self._valid_starts_buffers[0] = np.load(path + "buffer_one.npy")
        self._valid_starts_buffers[1] = np.load(path + "buffer_two.npy")
        self._valid_starts_buffer_usages[0] = (
            np.load(path + "usage_one_one.npy").item(),
            np.load(path + "usage_one_two.npy").item(),
        )
        self._valid_starts_buffer_usages[1] = (
            np.load(path + "usage_two_one.npy").item(),
            np.load(path + "usage_two_two.npy").item(),
        )
        self._current_buffer = np.load(path + "current_buffer.npy").item()

    def add(
        self,
        state,
        next_state,
        action,
        reward,
        done,
        sim_state,
        des_qv,
        des_qv_next,
        weight_mean,
        weight_std,
    ):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._sim_qps[self._ind, :] = sim_state[0]
        self._sim_qvs[self._ind, :] = sim_state[1]
        self._des_qps[self._ind, :] = des_qv[0]
        self._des_qvs[self._ind, :] = des_qv[1]
        self._des_qps_next[self._ind, :] = des_qv_next[0]
        self._des_qvs_next[self._ind, :] = des_qv_next[1]
        self._weight_means[self._ind, :] = weight_mean
        self._weight_stds[self._ind, :] = weight_std
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

    def sample_batch(self, batch_size, torch_batch=True, sequence=True):
        if torch_batch:
            batch = next(
                self.get_true_k_sequence_iter(
                    1, k=self._min_seq_len, batch_size=batch_size
                )
            ).to_torch_batch()
        else:
            batch = next(
                self.get_true_k_sequence_iter(
                    1, k=self._min_seq_len, batch_size=batch_size
                )
            )
        if not sequence:
            (
                states,
                next_states,
                actions,
                rewards,
                dones,
                sim_states,
                (des_qps, des_qvs),
                (des_qps_next, des_qvs_next),
                weight_means,
                weight_stds,
                idxs,
            ) = batch
            return (
                states[:, 0, :],
                next_states[:, 0, :],
                actions[:, 0, :],
                rewards[:, 0, :],
                dones[:, 0, :],
                (sim_states[0][:, 0, :], sim_states[1][:, 0, :]),
                (des_qps[:, 0, :], des_qvs[:, 0, :]),
                (des_qps_next[:, 0, :], des_qvs_next[:, 0, :]),
                weight_means[:, 0, :],
                weight_stds[:, 0, :],
                idxs,
            )
        else:
            return batch

    def update_des_qvs(self, idxes, des_qps, des_vs):
        des_qps, des_vs = to_np(des_qps), to_np(des_vs)
        self._des_qps[idxes, :] = des_qps[:, :-1, :]
        self._des_qvs[idxes, :] = des_vs[:, :-1, :]
        self._des_qps_next[idxes, :] = des_qps[..., 1:, :]
        self._des_qvs_next[idxes, :] = des_vs[..., 1:, :]

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
    def sim_qps(self):
        return self._sim_qps

    @property
    def sim_qvs(self):
        return self._sim_qvs

    @property
    def des_qps(self):
        return self._des_qps

    @property
    def des_qvs(self):
        return self._des_qvs

    @property
    def des_qps_next(self):
        return self._des_qps_next

    @property
    def des_qvs_next(self):
        return self._des_qvs_next

    @property
    def weight_means(self):
        return self._weight_means

    @property
    def weight_stds(self):
        return self._weight_stds

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
            return EnvSequence(
                self._buffer.states[full_trajectory_indices, :],
                self._buffer.next_states[full_trajectory_indices, :],
                self._buffer.actions[full_trajectory_indices, :],
                self._buffer.rewards[full_trajectory_indices],
                self._buffer.dones[full_trajectory_indices],
                (
                    self._buffer.sim_qps[full_trajectory_indices, :],
                    self._buffer.sim_qvs[full_trajectory_indices, :],
                ),
                (
                    self._buffer.des_qps[full_trajectory_indices, :],
                    self._buffer.des_qvs[full_trajectory_indices, :],
                ),
                (
                    self._buffer.des_qps_next[full_trajectory_indices, :],
                    self._buffer.des_qvs_next[full_trajectory_indices, :],
                ),
                self._buffer.weight_means[full_trajectory_indices, :],
                self._buffer.weight_stds[full_trajectory_indices, :],
                full_trajectory_indices,
            )
        else:
            raise StopIteration
