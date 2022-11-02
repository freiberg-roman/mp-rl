from pathlib import Path
from typing import Tuple

import numpy as np

from mprl.utils.buffer import EnvStep


class RandomRB:
    def __init__(self, capacity: int, state_dim, action_dim, sim_qp_dim, sim_qv_dim):
        self._capacity = 0
        self._max_capacity = capacity
        self._ind = 0
        self._s = np.empty((capacity, state_dim), dtype=np.float32)
        self._next_s = np.empty((capacity, state_dim), dtype=np.float32)
        self._acts = np.empty((capacity, action_dim), dtype=np.float32)
        self._rews = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=bool)
        self._sim_qps = np.empty((capacity, sim_qp_dim), dtype=np.float32)
        self._sim_qvs = np.empty((capacity, sim_qv_dim), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        sim_state: Tuple[np.ndarray, np.ndarray],
    ):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._sim_qps[self._ind, :] = sim_state[0]
        self._sim_qvs[self._ind, :] = sim_state[1]
        self._capacity = min(self._capacity + 1, self._max_capacity)
        self._ind = (self._ind + 1) % self._max_capacity

    def get_iter(self, it, batch_size):
        return RandomBatchIter(self, it, batch_size)

    def __getitem__(self, item):
        if 0 <= item < len(self):
            return EnvStep(
                self._s[item],
                self._next_s[item],
                self._acts[item],
                self._rews[item],
                self._dones[item],
                (self._sim_qps[item], self._sim_qvs[item]),
            )
        else:
            raise ValueError(
                "There are not enough time_steps stored to access this item"
            )

    def __len__(self):
        return self._capacity

    def save(self, base_path, folder):
        path = base_path + folder + "/rrb/"
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(path + "state.npy", self._s)
        np.save(path + "next_state.npy", self._next_s)
        np.save(path + "actions.npy", self._acts)
        np.save(path + "rewards.npy", self._rews)
        np.save(path + "dones.npy", self._dones)
        np.save(path + "qps.npy", self._qps)
        np.save(path + "qvs.npy", self._qvs)
        np.save(path + "capacity.npy", np.array([self._capacity], dtype=int))
        np.save(path + "index.npy", np.array([self._ind], dtype=int))

    def load(self, path):
        path = path + "/rrb/"
        self._s = np.load(path + "state.npy")
        self._next_s = np.load(path + "next_state.npy")
        self._acts = np.load(path + "actions.npy")
        self._rews = np.load(path + "rewards.npy")
        self._dones = np.load(path + "dones.npy")
        self._qps = np.load(path + "qps.npy")
        self._qvs = np.load(path + "qvs.npy")
        self._capacity = np.load(path + "capacity.npy").item()
        self._ind = np.load(path + "index.npy").item()

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
    def qps(self):
        return self._sim_qps

    @property
    def qvs(self):
        return self._sim_qvs


class RandomBatchIter:
    def __init__(self, buffer: RandomRB, it: int, batch_size: int):
        self._buffer: RandomRB = buffer
        self._it: int = it
        self._batch_size: int = batch_size
        self._current_it: int = 0

    def __iter__(self) -> "RandomBatchIter":
        return self

    def __next__(self) -> EnvStep:
        if self._current_it < self._it:
            idxs = np.random.randint(0, len(self._buffer), self._batch_size)
            self._current_it += 1
            return EnvStep(
                self._buffer.states[idxs],
                self._buffer.next_states[idxs],
                self._buffer.actions[idxs],
                self._buffer.rewards[idxs],
                self._buffer.dones[idxs],
            )
        else:
            raise StopIteration
