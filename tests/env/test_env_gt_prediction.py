import numpy as np
from omegaconf.omegaconf import OmegaConf
from mprl.env.mj_factory import create_mj_env
from mprl.models.physics.ground_truth import GroundTruth
from mprl.utils import RandomRB, SequenceRB


def buffer_sim_state_test(env_conf):
    env = create_mj_env(env_conf)
    state = env.reset(time_out_after=1000)
    buffer = RandomRB(OmegaConf.create({"capacity": 10000, "env": env_conf}))
    buffer_seq = SequenceRB(OmegaConf.create({"capacity": 10000, "env": env_conf}))
    gt_pred = GroundTruth(env_conf)

    for i in range(1000):
        action = env.sample_random_action()
        next_state, reward, done, timeout, sim_state = env.step(action)
        buffer.add(state, next_state, action, reward, done, sim_state)
        buffer_seq.add(state, next_state, action, reward, done, sim_state)
        state = next_state

        if (i + 1) % 25 == 0:
            buffer_seq.close_trajectory()

    assert timeout is True
    assert env.total_steps == 1000 == len(buffer)

    # sample batch and predict next state with gt prediction
    for batch in buffer.get_iter(
        it=1,
        batch_size=5,
    ):
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
        ) = batch.to_torch_batch()
        pred_next_states, _ = gt_pred.next_state(sim_states, actions)
    assert pred_next_states.shape == next_states.shape

    for batch in buffer_seq.get_iter(
        it=1,
        batch_size=5,
    ):
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
        ) = batch.to_torch_batch()
        pred_next_states, _ = gt_pred.next_state(sim_states, actions)
    assert pred_next_states.shape == next_states.shape

    for batch in buffer_seq.get_true_k_sequence_iter(
        it=1,
        k=6,
        batch_size=5,
    ):
        (
            states,
            next_states,
            actions,
            rewards,
            dones,
            sim_states,
        ) = batch.to_torch_batch()
        pred_next_states, _ = gt_pred.next_state(
            (sim_states[0][:, 0, :], sim_states[1][:, 0, :]), actions[:, 0, :]
        )
    assert pred_next_states.shape == next_states[:, 0, :].shape

def test_pred_reacher():
    env_conf = OmegaConf.create(
        {
            "name": "Reacher",
            "state_dim": 11,
            "action_dim": 2,
            "sim_qpos_dim": 4,
            "sim_qvel_dim": 4,
        }
    )
    buffer_sim_state_test(env_conf)

def test_pred_ant():
    env_conf = OmegaConf.create(
        {
            "name": "Ant",
            "state_dim": 27,
            "action_dim": 8,
            "sim_qpos_dim": 15,
            "sim_qvel_dim": 14,
        }
    )
    buffer_sim_state_test(env_conf)


def test_pred_half_cheetah():
    env_conf = OmegaConf.create(
        {
            "name": "HalfCheetah",
            "state_dim": 17,
            "action_dim": 6,
            "sim_qpos_dim": 9,
            "sim_qvel_dim": 9,
        }
    )
    buffer_sim_state_test(env_conf)


def test_pred_hopper():
    env_conf = OmegaConf.create(
        {
            "name": "Hopper",
            "state_dim": 11,
            "action_dim": 3,
            "sim_qpos_dim": 6,
            "sim_qvel_dim": 6,
        }
    )
    buffer_sim_state_test(env_conf)


def test_pred_humanoid():
    env_conf = OmegaConf.create(
        {
            "name": "Humanoid",
            "state_dim": 376,
            "action_dim": 17,
            "sim_qpos_dim": 24,
            "sim_qvel_dim": 23,
        }
    )
    buffer_sim_state_test(env_conf)


def test_pred_meta_button_press():
    env_conf = OmegaConf.create(
        {
            "name": "MetaButtonPress",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 10,
            "sim_qvel_dim": 10,
        }
    )
    buffer_sim_state_test(env_conf)


def test_pred_meta_push():
    env_conf = OmegaConf.create(
        {
            "name": "MetaPush",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 16,
            "sim_qvel_dim": 15,
        }
    )
    buffer_sim_state_test(env_conf)

def test_pred_meta_pick_and_place():
    env_conf = OmegaConf.create(
        {
            "name": "MetaPickAndPlace",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 16,
            "sim_qvel_dim": 15,
        }
    )
    buffer_sim_state_test(env_conf)

def test_pred_meta_reacher():
    env_conf = OmegaConf.create(
        {
            "name": "MetaReacher",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 16,
            "sim_qvel_dim": 15,
        }
    )
    buffer_sim_state_test(env_conf)

def test_pred_meta_window_close():
    env_conf = OmegaConf.create(
        {
            "name": "MetaWindowClose",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 10,
            "sim_qvel_dim": 10,
        }
    )
    buffer_sim_state_test(env_conf)

def test_pred_meta_window_open():
    env_conf = OmegaConf.create(
        {
            "name": "MetaWindowOpen",
            "state_dim": 39,
            "action_dim": 4,
            "sim_qpos_dim": 10,
            "sim_qvel_dim": 10,
        }
    )
    buffer_sim_state_test(env_conf)
