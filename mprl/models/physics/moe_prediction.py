import torch as ch

from mprl.utils.ds_helper import to_ts

from ..common.interfaces import Predictable, Trainable
from .moe import MixtureOfExperts


class MOEPrediction(Predictable, Trainable):
    def __init__(self, model: MixtureOfExperts, env_name: str, naive_prepare=False):
        self.model = model
        self.env_name = env_name
        self.naive_prepare = naive_prepare

    def next_state(self, states, actions, sim_states=None):
        states = to_ts(states)
        actions = to_ts(actions)
        p_states = self.prepare_state(states)
        next_states_delta = self.model.next_state_delta(p_states, actions)
        next_states = self.reconstruct_state(next_states_delta, states)
        return next_states, None

    def prepare_state(self, state):
        if self.env_name == "HalfCheetah":
            return state[..., 1:]
        elif "Meta" in self.env_name:
            if self.naive_prepare:
                return state
            else:
                return ch.cat((state[..., :18], state[..., -3:]), dim=-1)
        else:
            raise ValueError(f"Unknown environment name: {self.env_name}")

    def prepare_delta(self, delta):
        if self.env_name == "HalfCheetah":
            return delta
        elif "Meta" in self.env_name:
            if self.naive_prepare:
                return delta
            else:
                return ch.cat((delta[..., :18], delta[..., -3:]), dim=-1)
        else:
            raise ValueError(f"Unknown environment name: {self.env_name}")

    def reconstruct_state(self, prediction_delta, state):
        if self.env_name == "HalfCheetah":
            return prediction_delta + state
        elif "Meta" in self.env_name:
            if self.naive_prepare:
                return prediction_delta + state
            else:

                return ch.cat(
                    (
                        state[..., :18] + prediction_delta[..., :18],
                        state[..., :18],
                        state[..., -3:] + prediction_delta[..., -3:],
                    ),
                    dim=-1,
                )

    def update(self, batch=None):
        (states, next_states, actions, _, _) = batch.to_torch_batch()
        next_state_delta = next_states - states
        states = self.prepare_state(states)
        next_state_delta = self.prepare_delta(next_state_delta)
        update_loss = self.model.update(states, actions, next_state_delta)
        return update_loss

    def add_step(self, state, next_state, action, reward, done, sim_state):
        pass
