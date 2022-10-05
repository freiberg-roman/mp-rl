import math

import cv2
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
        self.should_record: bool = cfg.record_video
        self.record_each = cfg.record_each
        self.current_evaluation = 0
        self.record_mode = cfg.record_mode
        self.time_out_after: int = cfg.time_out_after
        self.images = []

    def evaluate(self, agent: Evaluable, after_performed_steps: int) -> dict:
        self.images = []
        to_log = {}

        total_reward = 0.0
        success: float = 0.0
        for i in range(self.num_eval_episodes):
            self.env.full_reset()
            agent.eval_reset()  # reset agent's internal state (e.g. motion primitives)

            state, sim_state = self.env.reset(time_out_after=self.time_out_after)
            done, time_out = False, False
            while not done and not time_out:
                action = agent.action_eval(state, sim_state)
                state, reward, done, time_out, sim_state, info = self.env.step(action)
                total_reward += reward

                # Only record the last episode if we are recording
                if self.should_record and i == self.num_eval_episodes - 1:
                    self.images.append(self.env.render(mode="rgb_array"))

            success += info.get("success", 0.0)

        if self.should_record and self.current_evaluation % self.record_each == 0:
            if self.record_mode == "disabled":
                out: cv2.VideoWriter = cv2.VideoWriter(
                    self.env.name + "_" + str(after_performed_steps) + ".avi",
                    cv2.VideoWriter_fourcc(*"DIVX"),
                    30,
                    (480, 480),
                )
                # save video
                for im in self.images:
                    out.write(im)
                out.release()
            elif self.record_mode == "online":
                imgs = np.transpose(np.array(self.images), (0, 3, 1, 2))
                to_log = {"video": wandb.Video(imgs, fps=math.floor(1 / self.env.dt))}

        avg_reward = total_reward / self.num_eval_episodes
        success_rate = success / self.num_eval_episodes
        self.current_evaluation += 1
        agent_log = agent.eval_log()
        return {
            **{
                "avg_episode_reward": avg_reward,
                "performance_steps": after_performed_steps,
                "success_rate": success_rate,
            },
            **agent_log,
            **to_log,
            **info,
        }
