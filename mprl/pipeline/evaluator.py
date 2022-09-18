from mprl.models.common import Evaluable

from .config_gateway import TrainConfigGateway


class Evaluator:
    def __init__(
        self,
        env,
        eval_config_gateway: TrainConfigGateway = Provider[
            containers.train_config_gateway
        ],
    ):
        self.num_eval_episodes = eval_config_gateway.get_eval_scheme().eval_episodes

    def evaluate(self, agent: Evaluable, training_steps: int):
        pass
