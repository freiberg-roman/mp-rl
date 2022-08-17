from omegaconf.omegaconf import OmegaConf
from mprl.env.mj_factory import create_mj_env
from mprl.models.physics.ground_truth import GroundTruth


def test_create_meta_pick_and_place():
    env = create_mj_env(OmegaConf.create({"name": "MetaPickAndPlace"}))
    env.reset(time_out_after=10)

    for _ in range(10):
        next_state, reward, done, timeout = env.step(env.sample_random_action())
    assert timeout is True
