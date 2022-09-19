import wandb
from dependency_injector import containers
from dependency_injector.wiring import Provider, inject
from tqdm import tqdm

from mprl.models.common import Actable, Trainable

from .config_gateway import TrainConfigGateway


class Trainer:
    @inject
    def __init__(
        self,
        env: MjEnv,
        train_config_gateway: TrainConfigGateway = Provider[
            containers.train_config_gateway
        ],
    ):
        self.env = env

        cfg = train_config_gateway.get_train_scheme()
        self.steps_per_epoch = cfg.steps_per_epoch
        self.total_steps = cfg.total_steps
        self.warm_steps = cfg.warm_steps
        self.update_each = cfg.update_each
        self.update_for = cfg.update_for
        assert (
            self.total_steps % self.iter_per_epoch == 0
            and self.total_steps > self.steps_per_epoch
        )

    def train_one_epoch(self, agent: Trainable & Actable) -> Trainable & Actable:
        """Train the agent for one epoch.

        :param agent: The agent to train.
        :return: The trained agent.
        """
        state, sim_state = self.env.reset()
        agent.sequence_reset()  # reset agent's internal state (e.g. motion primitives)
        for _ in tqdm(range(self.steps_per_epoch)):
            if self.env.total_steps < self.warm_steps:
                action, info = self.env.sample_random_action()
            else:
                action = agent.action(state, sim_state)

            next_state, reward, done, time_out = self.env.step(action)
            agent.add_step(state, next_state, action, reward, done, sim_state)

            state = next_state
            sim_state = self.env.get_state()

            if time_out or done:
                state, sim_state = self.env.reset()
                agent.sequence_reset()

            if self.env.total_steps % self.update_each == 0:
                for _ in range(self.update_for):
                    loggable = agent.update()
                wandb.log({**loggable, "update_step": self.env.total_steps})

        return agent

    @property
    def training_steps_left(self) -> bool:
        return self.env.total_steps < self.total_steps

    @property
    def performed_training_steps(self) -> int:
        return self.env.total_steps
