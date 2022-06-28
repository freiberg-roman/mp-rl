from copy import deepcopy

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from tqdm import tqdm

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.main.evaluate_agent import EvaluateAgent, EvaluateMPAgent
from mprl.models.sac import SAC
from mprl.models.sac_common import SACUpdate
from mprl.models.sac_mp import SACMP
from mprl.utils import RandomRB, RandomSequenceBasedRB
from mprl.utils.ds_helper import to_np


def train_sac(cfg_alg: DictConfig, cfg_env: DictConfig, cfg_wandb: DictConfig):
    env = create_mj_env(cfg_env)
    buffer = RandomRB(cfg_alg.buffer)
    agent = SAC(cfg_alg.agent)
    update = SACUpdate()
    eval = EvaluateAgent(cfg_env, record=cfg_wandb.record)
    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg_alg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg_alg.agent.lr)
    wandb.init(
        project=cfg_wandb.project,
        name=cfg_wandb.name + "_" + str(cfg_wandb.run_id),
        config=OmegaConf.to_container(deepcopy(cfg_alg)).update(
            OmegaConf.to_container(cfg_env)
        ),
        mode=cfg_wandb.mode,
    )

    # Training loop
    state = env.reset(time_out_after=cfg_env.time_out)
    total_reward = 0
    while env.total_steps < cfg_alg.train.total_steps:
        for i in tqdm(range(cfg_alg.train.steps_per_epoch)):
            if env.total_steps < cfg_alg.train.warm_start_steps:
                action = env.sample_random_action()
            else:
                action = agent.select_action(state)

            next_state, reward, done, time_out = env.step(action)
            buffer.add(state, next_state, action, reward, done)
            state = next_state
            total_reward += reward

            if time_out or done:
                total_reward = 0
                state = env.reset(time_out_after=cfg_env.time_out)

            if (
                len(buffer) < cfg_alg.train.warm_start_steps
            ):  # we first collect few sequences
                continue

            for batch in buffer.get_iter(
                it=cfg_alg.train.update_agent_every,
                batch_size=cfg_alg.train.batch_size,
            ):

                losses = update(agent, optimizer_policy, optimizer_critic, batch)
                wandb.log(losses)

        # Epoch end evaluation
        eval_results = eval(agent, performed_steps=env.total_steps)
        wandb.log(eval_results)


def train_mp_sac_vanilla(
    cfg_alg: DictConfig, cfg_env: DictConfig, cfg_wandb: DictConfig
):
    env = create_mj_env(cfg_env)
    buffer = RandomSequenceBasedRB(
        cfg_alg.buffer
    )  # in this case the next step corresponds to the next sequence
    agent = SACMP(cfg_alg.agent)
    update = SACUpdate()

    num_t = None if cfg_alg.agent.learn_time else cfg_alg.agent.time_steps
    eval = EvaluateMPAgent(cfg_env, record=cfg_wandb.record, num_t=num_t)

    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg_alg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg_alg.agent.lr)
    wandb.init(
        project=cfg_wandb.project,
        name=cfg_wandb.name + "_" + str(cfg_wandb.run_id),
        config=OmegaConf.to_container(deepcopy(cfg_alg)).update(
            OmegaConf.to_container(cfg_env)
        ),
        mode=cfg_wandb.mode,
    )
    mp_trajectory = MPTrajectory(cfg_alg.mp)
    pd_ctrl = PDController(cfg_alg.ctrl)

    state = env.reset()
    c_pos, c_vel = env.decompose(state)
    # Training loop
    while env.total_steps < cfg_alg.train.total_steps:
        for _ in tqdm(range(cfg_alg.train.steps_per_epoch)):
            with torch.no_grad():
                weight_time = agent.select_weights_and_time(state)
            mp_trajectory.re_init(weight_time, c_pos, c_vel, num_t=num_t)

            # Execute primitive
            acc_reward = 0.0
            for i, qv in enumerate(mp_trajectory):
                q, v = qv
                raw_action, logging_info = pd_ctrl.get_action(q, v, c_pos, c_vel)
                wandb.log(logging_info)
                action = to_np(raw_action)
                next_state, reward, done, _ = env.step(action)
                acc_reward += reward
                c_pos, c_vel = env.decompose(next_state)

            acc_reward /= mp_trajectory.steps_planned
            buffer.add(
                state,
                next_state,
                weight_time.cpu().detach().numpy(),
                acc_reward,
                done,
            )
            state = next_state

            if env.steps_after_reset > cfg_env.time_out:
                state = env.reset()
                c_pos, c_vel = env.decompose(state)

            # Perform one update step
            if (
                len(buffer) < cfg_alg.train.warm_start_steps
            ):  # we first collect few sequences
                continue
            for batch in buffer.get_iter(it=1, batch_size=cfg_alg.train.batch_size):
                losses = update(agent, optimizer_policy, optimizer_critic, batch)
                wandb.log(losses)

        # After epoch evaluation and saving
        eval_results = eval(agent, mp_trajectory, pd_ctrl, performed_steps=env.total_steps)
        wandb.log(eval_results)


def train_stepwise_mp_sac(cfg: OmegaConf):
    env = create_mj_env(cfg.env)
    time_out_after = cfg.env.time_out
    mp_trajectory = MPTrajectory(cfg.mp)
    pd_ctrl = PDController(cfg.ctrl)
    agent = MixedSAC(
        cfg.agent, planner=mp_trajectory, ctrl=pd_ctrl, decompose_state_fn=env.decompose
    )
    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg.agent.lr)
    buffer = RandomRB(cfg.buffer, use_bias=False)

    state = env.reset()
    while env.total_steps < cfg.train.total_steps:
        for _ in tqdm(range(cfg.train.steps_per_epoch)):
            if env.total_steps < cfg.train.warm_start_steps:
                action = env.sample_random_action()
            else:
                action = to_np(agent.select_action(state))
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, next_state, action, reward, done)
            state = next_state

            if env.steps_after_reset > cfg.env.time_out:
                state = env.reset()
            if (
                len(buffer) < cfg.train.warm_start_steps
            ):  # we first collect few sequences
                continue

            # Perform one update step
            for batch in buffer.get_iter(it=1, batch_size=cfg.train.batch_size):
                train_mdp_sac(agent, optimizer_policy, optimizer_critic, batch)

        # Evaluate
        evaluate_agent(
            cfg.env,
            agent,
            env.total_steps,
            time_out_after,
            record=True,
            use_wandb=False,
        )

        evaluate_mp_agent(
            cfg.env,
            agent,
            mp_trajectory,
            pd_ctrl,
            env.total_steps,
            time_out_after,
            record=True,
            use_wandb=False,
            num_t=5,
        )
