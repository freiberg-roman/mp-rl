from pathlib import Path

import numpy as np

from mprl.utils.buffer import EnvStep, EnvSteps
from mprl.utils.buffer.replay_buffer import ReplayBuffer


class RandomRB(ReplayBuffer):
    def __init__(self, cfg):
        self._cfg = cfg
        self._capacity = 0
        self._ind = 0
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)

    def add(self, state, next_state, action, reward, done):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        self._capacity = min(self._capacity + 1, self._cfg.capacity)
        self._ind = (self._ind + 1) % self._cfg.capacity

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
        np.save(path + "capacity.npy", np.array([self._capacity], dtype=int))
        np.save(path + "index.npy", np.array([self._ind], dtype=int))

    def load(self, path):
        path = path + "/rrb/"
        self._s = np.load(path + "state.npy")
        self._next_s = np.load(path + "next_state.npy")
        self._acts = np.load(path + "actions.npy")
        self._rews = np.load(path + "rewards.npy")
        self._dones = np.load(path + "dones.npy")
        self._capacity = np.load(path + "capacity.npy").item()
        self._ind = np.load(path + "index.npy").item()


class RandomSequenceBasedRB(RandomRB):
    def __init__(self, cfg):
        self._cfg = cfg
        self._capacity = 0
        self._ind = 0
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        cfg.env.action_dim = eval(cfg.env.action_dim) + 1  # add one for time action

        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)


class RandomBatchIter:
    def __init__(self, buffer: RandomRB, it: int, batch_size: int):
        self._buffer: RandomRB = buffer
        self._it: int = it
        self._batch_size: int = batch_size
        self._current_it: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it:
            idxs = np.random.randint(0, len(self._buffer), self._batch_size)
            self._current_it += 1
            return EnvSteps(
                self._buffer._s[idxs],
                self._buffer._next_s[idxs],
                self._buffer._acts[idxs],
                self._buffer._rews[idxs],
                self._buffer._dones[idxs],
            )
        else:
            raise StopIteration
