from copy import deepcopy

import numpy as np
import torch
import wandb
from tqdm import tqdm


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
    c_pos, c_vel = env.decompose(np.expand_dims(state, axis=0))
    # Training loop
    while env.total_steps < cfg_alg.train.total_steps:
        for _ in tqdm(range(cfg_alg.train.steps_per_epoch)):
            with torch.no_grad():
                weight_time, info = agent.select_weights_and_time(state)
                info.update(
                    {
                        "planned_at": env.total_steps,
                    }
                )
                wandb.log(info)
            mp_trajectory.re_init(weight_time, c_pos, c_vel, num_t=num_t)

            # Execute primitive
            acc_reward = 0.0
            for q, v in mp_trajectory:
                raw_action, logging_info = pd_ctrl.get_action(q, v, c_pos, c_vel)
                action = to_np(raw_action)
                next_state, reward, done, _ = env.step(action)
                acc_reward += reward
                c_pos, c_vel = env.decompose(np.expand_dims(next_state, axis=0))

            acc_reward /= mp_trajectory.steps_planned
            buffer.add(
                state,
                next_state,
                weight_time.squeeze().cpu().detach().numpy(),
                acc_reward,
                done,
            )
            state = next_state

            if env.steps_after_reset > cfg_env.time_out:
                state, sim_state = env.reset()
                c_pos, c_vel = env.decompose(np.expand_dims(state, axis=0))

            # Perform one update step
            if (
                len(buffer) < cfg_alg.train.warm_start_steps
            ):  # we first collect few sequences
                continue
            for batch in buffer.get_iter(it=1, batch_size=cfg_alg.train.batch_size):
                losses = update(agent, optimizer_policy, optimizer_critic, batch)
                wandb.log(losses)

        # After epoch evaluation and saving
        eval_results = eval(
            agent, mp_trajectory, pd_ctrl, performed_steps=env.total_steps
        )
        wandb.log(eval_results)


def train_stepwise_mp_sac(
    cfg_alg: DictConfig, cfg_env: DictConfig, cfg_wandb: DictConfig
):
    env = create_mj_env(cfg_env)
    buffer = RandomRB(cfg_alg.buffer)
    mp_trajectory = MPTrajectory(cfg_alg.mp)
    pd_ctrl = PDController(cfg_alg.ctrl)
    agent = SACMixed(
        cfg_alg.agent, mp_trajectory, pd_ctrl, decompose_state_fn=env.decompose
    )

    # Compose update function
    if cfg_alg.prediction.name == "GroundTruth":
        model: Prediction = GroundTruth(cfg_env)
    elif cfg_alg.prediction.name == "MixtureOfExperts":
        model = MixtureOfExperts(cfg_alg.prediction)
    else:
        raise ValueError("Unknown model")

    if cfg_alg.reward_weighting == "mean":
        policy_loss = MixedMeanSACModelPolicyLoss(model)
    elif cfg_alg.reward_weighting == "likelihood":
        policy_loss = MixedWeightedSACModelPolicyLoss(model)
    else:
        raise ValueError("Unknown reward weighting")
    update = SACUpdate(policy_loss=policy_loss, critic_loss=sac_mp_critic_loss)

    num_t = cfg_alg.agent.time_steps
    eval_mp = EvaluateMPAgent(cfg_env, record=cfg_wandb.record, num_t=num_t)
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

    state, sim_state = env.reset()
    while env.total_steps < cfg_alg.train.total_steps:
        for _ in tqdm(range(cfg_alg.train.steps_per_epoch)):
            if env.total_steps < cfg_alg.train.warm_start_steps:
                action = env.sample_random_action()
            else:
                raw_action, _ = agent.select_action(
                    state, (sim_state[0][None], sim_state[1][None])
                )
                action = np.ravel(to_np(raw_action))
            next_state, reward, done, _, sim_state = env.step(action)
            buffer.add(state, next_state, action, reward, done, sim_state)
            state = next_state

            if env.steps_after_reset > cfg_env.time_out:
                state, sim_state = env.reset()
                agent.replan()
            if (
                len(buffer) < cfg_alg.train.warm_start_steps
            ):  # we first collect few sequences
                continue

            # Perform one update step
            for batch in buffer.get_iter(it=1, batch_size=cfg_alg.train.batch_size):
                update(agent, optimizer_policy, optimizer_critic, batch)

        # Evaluate
        eval_mp_results = eval_mp(
            agent, mp_trajectory, pd_ctrl, performed_steps=env.total_steps
        )
        wandb.log(eval_mp_results)
        eval_results = eval(agent, performed_steps=env.total_steps)
        wandb.log(eval_results)


def train_stepwise_mp_sac_offpolicy(
    cfg_alg: DictConfig, cfg_env: DictConfig, cfg_wandb: DictConfig
):
    env = create_mj_env(cfg_env)
    buffer = SequenceRB(cfg_alg.buffer)
    mp_trajectory = MPTrajectory(cfg_alg.mp)
    pd_ctrl = PDController(cfg_alg.ctrl)
    agent = SACMixed(
        cfg_alg.agent, mp_trajectory, pd_ctrl, decompose_state_fn=env.decompose
    )

    # Compose update function
    if cfg_alg.reward_weighting == "mean":
        policy_loss = MixedMeanSACOffPolicyLoss()
    elif cfg_alg.reward_weighting == "likelihood":
        policy_loss = MixedWeightedSACOffPolicyLoss()
    else:
        raise ValueError("Unknown reward weighting")
    critic_loss = sac_critic_loss_sequenced
    update = SACUpdate(policy_loss=policy_loss, critic_loss=critic_loss)

    num_t = cfg_alg.agent.time_steps
    eval_mp = EvaluateMPAgent(cfg_env, record=cfg_wandb.record, num_t=num_t)
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

    state = env.reset()
    while env.total_steps < cfg_alg.train.total_steps:
        for _ in tqdm(range(cfg_alg.train.steps_per_epoch)):
            if env.total_steps < cfg_alg.train.warm_start_steps:
                action = env.sample_random_action()
            else:
                raw_action, _ = agent.select_action(state)
                action = to_np(raw_action)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, next_state, action, reward, done)
            state = next_state

            if env.steps_after_reset > cfg_env.time_out:
                state = env.reset()
                agent.replan()
                buffer.close_trajectory()
            if (
                len(buffer) < cfg_alg.train.warm_start_steps
            ):  # we first collect few sequences
                continue

            # Perform one update step
            for batch in buffer.get_true_k_sequence_iter(
                it=1, k=num_t, batch_size=cfg_alg.train.batch_size
            ):
                update(agent, optimizer_policy, optimizer_critic, batch)

        # Evaluate
        eval_mp_results = eval_mp(
            agent, mp_trajectory, pd_ctrl, performed_steps=env.total_steps
        )
        wandb.log(eval_mp_results)
        eval_results = eval(agent, performed_steps=env.total_steps)
        wandb.log(eval_results)
