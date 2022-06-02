from omegaconf import OmegaConf

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.models import MDPSAC
from mprl.utils import SequenceRB


def train_sac(cfg: OmegaConf):
    pass


def train_mp_sac_vanilla(cfg: OmegaConf):
    env = create_mj_env(cfg.env)
    mpsac_agent = MDPSAC(cfg.agent)
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


def train_mp_sac_virtual(cfg: OmegaConf):
    pass


def train_mp_sac_augmented(cfg: OmegaConf):
    pass


def train_stepwise_mp_sac(cfg: OmegaConf):
    pass
