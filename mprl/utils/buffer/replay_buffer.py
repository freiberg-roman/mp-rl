from abc import ABC, abstractmethod

from mprl.utils import EnvStep, EnvSteps


class ReplayBuffer(ABC):
    @abstractmethod
    def add(self, state, next_state, action, reward, done):
        pass

    def add_step(self, time_step: EnvStep):
        self.add(
            time_step.state,
            time_step.next_state,
            time_step.action,
            time_step.reward,
            time_step.done,
        )

    @abstractmethod
    def add_batch(self, states, next_states, actions, rewards, dones):
        pass

    def add_batch_steps(self, time_steps: EnvSteps):
        self.add_batch(
            time_steps.states,
            time_steps.next_states,
            time_steps.actions,
            time_steps.rewards,
            time_steps.dones,
        )

    @abstractmethod
    def get_iter(self, it, batch_size):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        return 0
