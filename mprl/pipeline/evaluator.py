import numpy as np
import wandb

from mprl.env import MPRLEnvironment
from mprl.models.common import Evaluable

from .config_gateway import TrainConfigGateway


class Evaluator:
    def __init__(self, env: MPRLEnvironment, eval_config_gateway: TrainConfigGateway):
        cfg = eval_config_gateway.get_evaluation_config()
        self.num_eval_episodes: int = cfg.num_eval_episodes
        self.env = env
        self.time_out_after: int = cfg.time_out_after

    def evaluate(
        self, agent: Evaluable, after_performed_steps: int, render=False
    ) -> dict:
        to_log = {}

        total_reward = 0.0
        success: float = 0.0
        for i in range(self.num_eval_episodes):
            self.env.full_reset()
            agent.eval_reset()  # reset agent's internal state (e.g. motion primitives)

            state, sim_state = self.env.reset(time_out_after=self.time_out_after)
            done, time_out = False, False
            successes = []
            actions = []
            while not done and not time_out:
                action = agent.action_eval(state, sim_state)
                state, reward, done, time_out = self.env.step(action)
                if render:
                    self.env.render()

                total_reward += reward
                info = self.env.get_info()
                successes.append(info.get("success", -1.0))
                actions.append(action)

            success += float(max(successes))

        avg_reward = total_reward / self.num_eval_episodes
        success_rate = success / self.num_eval_episodes
        agent_log = agent.eval_log()
        return {
            **{
                "avg_episode_reward": avg_reward,
                "performance_steps": after_performed_steps,
                "success_rate": success_rate,
                "action_eval_histogram": wandb.Histogram(np.array(actions).flatten()),
            },
            **agent_log,
            **to_log,
            **info,
        }
