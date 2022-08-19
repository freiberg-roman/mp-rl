import numpy as np
from omegaconf.omegaconf import OmegaConf
from mprl.env.mj_factory import create_mj_env
from mprl.models.physics.ground_truth import GroundTruth
from mprl.utils import RandomRB


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
    env = create_mj_env(env_conf)
    state = env.reset(time_out_after=1000)
    buffer = RandomRB(OmegaConf.create({"capacity": 10000, "env": env_conf}))
    gt_pred = GroundTruth(env_conf)

    for _ in range(1000):
        action = env.sample_random_action()
        next_state, reward, done, timeout, sim_state = env.step(action)
        buffer.add(state, next_state, action, reward, done, sim_state)
        state = next_state

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
        pred_next_states = gt_pred.next_state(sim_states, actions)
    assert pred_next_states.shape == next_states.shape
