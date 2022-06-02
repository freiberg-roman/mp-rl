from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.models import MDPSAC
from mprl.models.sac import train_mdp_sac
from mprl.utils import RandomRB, SequenceRB


def train_sac(cfg: OmegaConf):
    env = create_mj_env(cfg.env)
    buffer = RandomRB(cfg.buffer)
    agent = MDPSAC(cfg.agent)
    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg.agent.lr)
    time_out_after = cfg.env.time_out

    state = env.reset(time_out_after=time_out_after)
    total_reward = 0
    while env.total_steps < cfg.train.total_steps:
        # Train
        for i in tqdm(range(cfg.train.steps_per_epoch)):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, next_state, action, reward, done)
            state = next_state
            total_reward += reward

            if done:
                print(
                    "Total episode reward: ",
                    total_reward,
                    " after ",
                    env.total_steps,
                    " steps.",
                )
                total_reward = 0
                state = env.reset(time_out_after=time_out_after)

            if i % cfg.train.update_agent_every == 0:
                for batch in buffer.get_iter(
                    it=cfg.train.update_agent_every, batch_size=cfg.train.batch_size
                ):
                    train_mdp_sac(agent, optimizer_policy, optimizer_critic, batch)

        # Evaluate each epoch
        env_eval = create_mj_env(cfg.env)
        state = env_eval.reset()
        eval_reward = 0
        for _ in range(time_out_after):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _ = env_eval.step(action)
            eval_reward += reward
            env_eval.render(mode="human")
        print(
            "Total evaluation reward: ",
            eval_reward,
            " after ",
            env.total_steps,
            " steps.",
        )
        env_eval.close()
        del env_eval

        # Save each epoch this model


def train_mp_sac_vanilla(cfg: OmegaConf):
    env = create_mj_env(cfg.env)
    mpsac_agent = MDPSAC(cfg.agent)
    buffer = RandomRB(
        cfg.buffer
    )  # in this case the next step corresponds to the next sequence
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
