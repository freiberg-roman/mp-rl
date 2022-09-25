import cv2

from mprl.env import MPRLEnvironment
from mprl.models.common import Evaluable

from .config_gateway import TrainConfigGateway


class Evaluator:
    def __init__(self, env: MPRLEnvironment, eval_config_gateway: TrainConfigGateway):
        cfg = eval_config_gateway.get_evaluation_config()
        self.num_eval_episodes: int = cfg.num_eval_episodes
        self.env = env
        self.should_record: bool = cfg.record_video
        self.time_out_after: int = cfg.time_out_after
        self.images = []

    def evaluate(self, agent: Evaluable, after_performed_steps: int) -> dict:
        self.images = []

        for i in range(self.num_eval_episodes):
            self.env.full_reset()
            agent.eval_reset()  # reset agent's internal state (e.g. motion primitives)

            total_reward = 0.0
            state, sim_state = self.env.reset(time_out_after=self.time_out_after)
            done, time_out = False, False
            while not done and not time_out:
                action = agent.action_eval(state, sim_state)
                state, reward, done, time_out, sim_state, _ = self.env.step(action)
                total_reward += reward

                # Only record the last episode if we are recording
                if self.should_record and i == self.num_eval_episodes - 1:
                    self.images.append(self.env.render(mode="rgb_array"))

        if self.should_record:
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

        avg_reward = total_reward / self.num_eval_episodes
        return {"avg_episode_reward": avg_reward}
