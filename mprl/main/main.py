import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.models import SAC
from mprl.utils import SequenceRB


@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):

    if cfg.mode == "train_mprl":
        env = create_mj_env(cfg.env)
        mpsac_agent = SAC(cfg.agent)
        buffer = SequenceRB(cfg.buffer)
        mp_trajectory = MPTrajectory(cfg.mp)
        pd_ctrl = PDController(cfg.ctrl)

        state = env.reset()
        c_pos, c_vel = env.decompose(state)
        while env.total_steps < cfg.train.total_env_steps:
            for _ in range(cfg.train.steps_per_epoch):
                mean, L, time = mpsac_agent.select_action(state)
                mp_trajectory.re_init(mean, L, time, c_pos, c_vel)

                # Execute primitive
                for q, v in mp_trajectory:
                    action = pd_ctrl.ctrl(q, v, c_pos, c_vel, bias=env.get_forces())
                    next_state, reward, done, _ = env.step(action)
                    c_pos, c_vel = env.decompose(next_state)
                    buffer.add(
                        state, next_state, action, reward, done
                    )  # TODO action on sequences

                if env.steps_after_reset > cfg.env.reset_after:
                    state = env.reset()
                    c_pos, c_vel = env.decompose(state)

            # After epoch evaluation and saving


if __name__ == "__main__":
    run()
