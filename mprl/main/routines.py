import cv2
import torch
import wandb
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm

from mprl.controllers import MPTrajectory, PDController
from mprl.env import create_mj_env
from mprl.models import MDPSAC, OMDPSAC, MixedSAC
from mprl.models.sac import train_mdp_sac
from mprl.utils import RandomRB, RandomSequenceBasedRB
from mprl.utils.ds_helper import to_np


@torch.no_grad()
def evaluate_agent(
    env_name, agent, total_training_steps, time_out_after, record=False, use_wandb=False
):
    env_eval = create_mj_env(env_name)
    state = env_eval.reset(time_out_after=time_out_after)
    eval_reward = 0
    images = []
    for _ in range(time_out_after):
        action = to_np(agent.select_action(state, evaluate=True))
        state, reward, done, time_out = env_eval.step(action)
        eval_reward += reward
        if record:
            images.append(env_eval.render(mode="rgb_array"))
        if done or time_out:
            break
    print(
        "Total evaluation reward: ",
        eval_reward,
        " after ",
        total_training_steps,
        " steps.",
    )
    if use_wandb:
        wandb.log({"total_reward": eval_reward, "total_steps": total_training_steps})
    env_eval.close()
    del env_eval

    out: cv2.VideoWriter = cv2.VideoWriter(
        "hc_video_time.avi", cv2.VideoWriter_fourcc(*"DIVX"), 60, (480, 480)
    )
    # save video
    for im in images:
        out.write(im)
    out.release()


@torch.no_grad()
def evaluate_mp_agent(
    env_name,
    agent,
    mp_trajectory,
    ctrl,
    total_training_steps,
    time_out_after,
    record=False,
    use_wandb=False,
    num_t=None,
):
    env_eval = create_mj_env(env_name)
    state = env_eval.reset(time_out_after=time_out_after)
    eval_reward = 0
    c_pos, c_vel = env_eval.decompose(state)
    images = []
    for _ in range(time_out_after):
        weight_time = agent.select_weights_and_time(state)
        mp_trajectory.re_init(weight_time, c_pos, c_vel, num_t=num_t)

        # Execute primitive
        for q, v in mp_trajectory:
            action = to_np(ctrl.get_action(q, v, c_pos, c_vel))
            next_state, reward, done, time_out = env_eval.step(action)
            eval_reward += reward
            c_pos, c_vel = env_eval.decompose(next_state)
            if record:
                images.append(env_eval.render(mode="rgb_array"))
            if done or time_out:
                break
        state = next_state

        if done or time_out:
            break
    print(
        "Total evaluation reward: ",
        eval_reward,
        " after ",
        total_training_steps,
        " steps.",
    )
    if use_wandb:
        wandb.log({"total_reward": eval_reward, "total_steps": total_training_steps})
    env_eval.close()
    del env_eval
    out: cv2.VideoWriter = cv2.VideoWriter(
        "hc_mp_video.avi", cv2.VideoWriter_fourcc(*"DIVX"), 60, (480, 480)
    )
    # save video
    for im in images:
        out.write(im)
    out.release()


def train_sac(cfg: OmegaConf):
    env = create_mj_env(cfg.env)
    buffer = RandomRB(cfg.buffer)
    agent = MDPSAC(cfg.agent)
    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg.agent.lr)
    time_out_after = cfg.env.time_out

    state = env.reset(time_out_after=time_out_after)
    total_reward = 0
    if cfg.use_wandb:
        wandb.init(
            project="reacher",
            name=cfg.env.name + "_" + str(cfg.run),
            config=OmegaConf.to_container(cfg),
        )
        training_steps = 0

    while env.total_steps < cfg.train.total_steps:
        # Train
        for i in tqdm(range(cfg.train.steps_per_epoch)):
            if env.total_steps < cfg.train.warm_start_steps:
                action = env.sample_random_action()
            else:
                action = agent.select_action(state)

            next_state, reward, done, time_out = env.step(action)
            buffer.add(state, next_state, action, reward, done)
            state = next_state
            total_reward += reward

            if time_out or done:
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
                    q1_loss, q2_loss, pi_loss, _, _ = train_mdp_sac(
                        agent, optimizer_policy, optimizer_critic, batch
                    )
                    wandb.log(
                        {
                            "q1_loss": q1_loss,
                            "q2_loss": q2_loss,
                            "pi_loss": pi_loss,
                            "training_steps": training_steps,
                        }
                    )
                    training_steps += 1

        # Evaluate
        evaluate_agent(
            cfg.env,
            agent,
            env.total_steps,
            time_out_after,
            record=True,
            use_wandb=cfg.use_wandb,
        )


def train_mp_sac_vanilla(cfg: OmegaConf):
    use_wandb = cfg.use_wandb
    if use_wandb:
        wandb.init(
            project="reacher_mp_sac",
            name=cfg.env.name + "_" + str(cfg.run),
            config=OmegaConf.to_container(cfg),
        )
        # wandb_counter = {}

    env = create_mj_env(cfg.env)
    time_out_after = cfg.env.time_out
    agent = OMDPSAC(cfg.agent, use_wandb=use_wandb)
    optimizer_policy = Adam(agent.policy.parameters(), lr=cfg.agent.lr)
    optimizer_critic = Adam(agent.critic.parameters(), lr=cfg.agent.lr)
    buffer = RandomSequenceBasedRB(
        cfg.buffer
    )  # in this case the next step corresponds to the next sequence
    num_t = None if cfg.agent.learn_time else cfg.agent.time_steps
    mp_trajectory = MPTrajectory(cfg.mp, use_wandb=use_wandb)
    pd_ctrl = PDController(cfg.ctrl, use_wandb=use_wandb)

    state = env.reset()
    c_pos, c_vel = env.decompose(state)
    while env.total_steps < cfg.train.total_steps:
        for _ in tqdm(range(cfg.train.steps_per_epoch)):
            with torch.no_grad():
                weight_time = agent.select_weights_and_time(state)
            mp_trajectory.re_init(weight_time, c_pos, c_vel, num_t=num_t)

            # Execute primitive
            acc_reward = 0.0
            discount = 1.0
            for i, qv in enumerate(mp_trajectory):
                q, v = qv
                action = to_np(pd_ctrl.get_action(q, v, c_pos, c_vel))
                next_state, reward, done, _ = env.step(action)
                acc_reward += discount**i * reward
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

            if env.steps_after_reset > cfg.env.time_out:
                state = env.reset()
                c_pos, c_vel = env.decompose(state)

            # Perform one update step
            if (
                len(buffer) < cfg.train.warm_start_steps
            ):  # we first collect few sequences
                continue
            for batch in buffer.get_iter(it=1, batch_size=cfg.train.batch_size):
                q1_loss, q2_loss, pi_loss, _, _ = train_mdp_sac(
                    agent, optimizer_policy, optimizer_critic, batch
                )
                if use_wandb:
                    wandb.log(
                        {"q1_loss": q1_loss, "q2_loss": q2_loss, "pi_loss": pi_loss}
                    )

        # After epoch evaluation and saving
        evaluate_mp_agent(
            cfg.env,
            agent,
            mp_trajectory,
            pd_ctrl,
            env.total_steps,
            time_out_after,
            record=True,
            use_wandb=cfg.use_wandb,
            num_t=num_t,
        )


def train_mp_sac_virtual(cfg: OmegaConf):
    pass


def train_mp_sac_augmented(cfg: OmegaConf):
    pass


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
