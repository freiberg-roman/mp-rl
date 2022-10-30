from abc import ABC

from mprl.models import Predictable, Trainable
from mprl.models.physics.moe import MixtureOfExperts
from mprl.utils.ds_helper import to_np, to_ts


class MOEPrediction(Predictable, Trainable):
    def __init__(self, model: MixtureOfExperts, env_name: str):
        self.model = model
        self.env_name = env_name

    def next_state(self, states, sim_states, actions):
        states = to_ts(states)
        actions = to_ts(actions)
        states = self.prepare_state(states)
        next_states = self.model.next_state(states, actions)
        next_states = self.reconstruct_state(next_states, states)
        next_states = to_np(next_states)
        return next_states

    def prepare_state(self, state):
        if self.env_name == "HalfCheetah":
            return state[:, 1:]

    def reconstruct_state(self, prediction, state):
        if self.env_name == "HalfCheetah":
            return prediction

    def update(self, batch=None):
        (states, next_states, actions, _, _, _) = batch.to_torch_batch()
        states = self.prepare_state(states)
        next_states = self.prepare_state(next_states)
        update_loss = self.model.update(states, actions, next_states)
        return update_loss

    def add_step(self, state, next_state, action, reward, done, sim_state):
        pass
