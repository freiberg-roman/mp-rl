from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from mprl.utils.buffer import EnvStep, EnvSteps, EnvStepsWithBias
from mprl.utils.buffer.replay_buffer import ReplayBuffer


class RandomRB(ReplayBuffer):
    def __init__(self, cfg, use_bias=False):
        self._cfg = cfg
        self._capacity = 0
        self._ind = 0
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._use_bias = use_bias
        if use_bias:
            self._bias = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)

    def add(self, state, next_state, action, reward, done, bias=None):
        self._s[self._ind, :] = state
        self._next_s[self._ind, :] = next_state
        self._acts[self._ind, :] = action
        self._rews[self._ind] = reward
        self._dones[self._ind] = done
        if bias is not None:
            self._bias[self._ind, :] = bias

        self._capacity = min(self._capacity + 1, self._cfg.capacity)
        self._ind = (self._ind + 1) % self._cfg.capacity

    def add_batch(self, states, next_states, actions, rewards, dones, biases=None):
        length_batch = len(states)
        start_ind = self._ind
        end_ind = min(start_ind + length_batch, self._cfg.capacity)
        stored_ind = end_ind - start_ind

        self._s[start_ind:end_ind, :] = states[:stored_ind]
        self._next_s[start_ind:end_ind, :] = next_states[:stored_ind]
        self._acts[start_ind:end_ind, :] = actions[:stored_ind]
        self._rews[start_ind:end_ind] = rewards[:stored_ind]
        self._dones[start_ind:end_ind] = dones[:stored_ind]
        if biases is not None:
            self._bias[start_ind:end_ind] = biases

        if start_ind + length_batch > self._cfg.capacity:
            self._ind = 0
            self._capacity = self._cfg.capacity
            self.add_batch(
                states[stored_ind:, :],
                next_states[stored_ind:, :],
                actions[stored_ind:, :],
                rewards[stored_ind:],
                dones[stored_ind:],
            )
        else:
            self._ind = self._ind + length_batch
            self._capacity = max(self._capacity, self._ind)

    def get_iter(self, it, batch_size):
        return RandomBatchIter(self, it, batch_size, use_bias=self._use_bias)

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
        if self._use_bias:
            np.save(path + "biases.npy", self._dones)
        np.save(path + "capacity.npy", np.array([self._capacity], dtype=int))
        np.save(path + "index.npy", np.array([self._ind], dtype=int))

    def load(self, path):
        path = path + "/rrb/"
        self._s = np.load(path + "state.npy")
        self._next_s = np.load(path + "next_state.npy")
        self._acts = np.load(path + "actions.npy")
        self._rews = np.load(path + "rewards.npy")
        self._dones = np.load(path + "dones.npy")
        if self._use_bias:
            self._bias = np.load(path + "biases.npy")
        self._capacity = np.load(path + "capacity.npy").item()
        self._ind = np.load(path + "index.npy").item()


class RandomSequenceBasedRB(RandomRB):
    def __init__(self, cfg):
        self._cfg = cfg
        self._capacity = 0
        self._ind = 0
        self._s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        self._next_s = np.empty((cfg.capacity, cfg.env.state_dim), dtype=np.float32)
        cfg.env.action_dim = eval(cfg.env.action_dim)
        self._acts = np.empty((cfg.capacity, cfg.env.action_dim), dtype=np.float32)
        self._rews = np.empty(cfg.capacity, dtype=np.float32)
        self._dones = np.empty(cfg.capacity, dtype=bool)


class RandomValidationRB(ReplayBuffer):
    def __init__(self, cfg, val_percentage):
        self._train_buffer = RandomRB(cfg)
        self._val_buffer = RandomRB(cfg)
        self._val_percentage = val_percentage

    def __len__(self):
        return len(self._val_buffer) + len(self._train_buffer)

    def get_iter(self, it, batch_size):
        return self._train_buffer.get_iter(it, batch_size), self._val_buffer.get_iter(
            it, batch_size
        )

    def add(self, state, next_state, action, reward, done):
        if (
            len(self._train_buffer) == 0
            or len(self._val_buffer) / len(self) >= self._val_percentage
        ):
            self._train_buffer.add(state, next_state, action, reward, done)
        else:
            self._val_buffer.add(state, next_state, action, reward, done)

    def add_batch(self, states, next_states, actions, rewards, dones):
        if (
            len(self._train_buffer) == 0
            or len(self._val_buffer) / len(self) >= self._val_percentage
        ):
            self._train_buffer.add_batch(states, next_states, actions, rewards, dones)
        else:
            self._val_buffer.add_batch(states, next_states, actions, rewards, dones)

    def __getitem__(self, item):
        if item > len(self._train_buffer):
            return self._val_buffer[item - len(self._train_buffer)]
        else:
            return self._train_buffer[item]

    @property
    def train_buffer(self):
        return self._train_buffer

    @property
    def val_buffer(self):
        return self._val_buffer


class RandomBatchIter:
    def __init__(self, buffer: RandomRB, it: int, batch_size: int, use_bias: bool):
        self._buffer = buffer
        self._it = it
        self._batch_size = batch_size
        self._current_it = 0
        self._use_bias = use_bias

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_it < self._it:
            idxs = np.random.randint(0, len(self._buffer), self._batch_size)
            self._current_it += 1
            if self._use_bias:
                return EnvStepsWithBias(
                    self._buffer._s[idxs],
                    self._buffer._next_s[idxs],
                    self._buffer._acts[idxs],
                    self._buffer._rews[idxs],
                    self._buffer._dones[idxs],
                    self._buffer._bias[idxs],
                )
            else:
                return EnvSteps(
                    self._buffer._s[idxs],
                    self._buffer._next_s[idxs],
                    self._buffer._acts[idxs],
                    self._buffer._rews[idxs],
                    self._buffer._dones[idxs],
                )
        else:
            raise StopIteration


class AllKSequenceIter:
    def __init__(self, buffer: RandomRB, it: int):
        self._buffer = buffer
        self._it = it
        self._current_it = 0

    def __iter__(self):
        pass

    def __next__(self):
        pass
