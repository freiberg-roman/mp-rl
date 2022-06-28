from typing import Optional, Union

import cv2
import torch
from omegaconf import DictConfig

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.env.mujoco.mj_env import MujocoEnv
from mprl.models.sac import SAC
from mprl.models.sac_mixed.agent import SACMixed
from mprl.models.sac_mp.agent import SACMP
from mprl.utils.ds_helper import to_np


class EvaluateAgent:
    def __init__(self, cfg_env: DictConfig, record: Optional[bool] = False):
        self.cfg_env: MujocoEnv = cfg_env
        self.record: bool = record

    @torch.no_grad()
    def __call__(self, agent: Union[SAC, SACMixed], performed_steps: int = 0):
        env_eval = create_mj_env(self.cfg_env)
        images = []
        total_reward = 0

        state = env_eval.reset(time_out_after=self.cfg_env.time_out)
        done, time_out = False, False
        while not done and not time_out:
            action = to_np(agent.select_action(state, evaluate=True))
            state, reward, done, time_out = env_eval.step(action)
            total_reward += reward
            if self.record:
                images.append(env_eval.render(mode="rgb_array"))
        env_eval.close()
        del env_eval

        print(
            "Total episode reward: ",
            total_reward,
            " after ",
            performed_steps,
            " steps.",
        )

        if self.record:
            out: cv2.VideoWriter = cv2.VideoWriter(
                env_eval.name + "_" + str(performed_steps) + ".avi",
                cv2.VideoWriter_fourcc(*"DIVX"),
                60,
                (480, 480),
            )
            # save video
            for im in images:
                out.write(im)
            out.release()

        return {
            "total_reward": total_reward,
            "performed_steps": performed_steps,
        }


class EvaluateMPAgent:
    def __init__(
        self,
        cfg_env: DictConfig,
        record: Optional[bool] = False,
        num_t: Optional[int] = None,
    ):
        self.cfg_env: MujocoEnv = cfg_env
        self.record: bool = record
        self.num_t: int = num_t

    def __call__(
        self,
        agent: SACMP,
        trajectory_planner: MPTrajectory,
        controller: PDController,
        performed_steps: int = 0,
    ):
        env_eval = create_mj_env(self.cfg_env)
        images = []
        total_reward = 0

        state = env_eval.reset(time_out_after=self.cfg_env.time_out)
        done, time_out = False, False
        c_pos, c_vel = env_eval.decompose(state)
        while not done and not time_out:
            weight_time = agent.select_weights_and_time(state)
            trajectory_planner.re_init(weight_time, c_pos, c_vel, num_t=self.num_t)

            # Execute primitive
            for q, v in trajectory_planner:
                raw_action, _ = controller.get_action(q, v, c_pos, c_vel)
                action = to_np(raw_action)
                next_state, reward, done, time_out = env_eval.step(action)
                total_reward += reward
                c_pos, c_vel = env_eval.decompose(next_state)
                if self.record:
                    images.append(env_eval.render(mode="rgb_array"))
                if done or time_out:
                    break
            state = next_state

            if done or time_out:
                break

        print(
            "Total episode reward: ",
            total_reward,
            " after ",
            performed_steps,
            " steps.",
        )

        if self.record:
            out: cv2.VideoWriter = cv2.VideoWriter(
                env_eval.name + "_" + str(performed_steps) + ".avi",
                cv2.VideoWriter_fourcc(*"DIVX"),
                60,
                (480, 480),
            )
            # save video
            for im in images:
                out.write(im)
            out.release()

        return {
            "total_reward": total_reward,
            "performed_steps": performed_steps,
        }
