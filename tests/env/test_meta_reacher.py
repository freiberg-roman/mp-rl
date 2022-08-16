from omegaconf.omegaconf import OmegaConf
from mprl.env.mj_factory import create_mj_env
from mprl.models.physics.ground_truth import GroundTruth


def test_create_meta_reacher_env():
    env = create_mj_env(OmegaConf.create({"name": "MetaReacher"}))
    env.reset(time_out_after=10)

    for _ in range(10):
        next_state, reward, done, timeout = env.step(env.sample_random_action())
    assert timeout == True


def test_ground_truth_prediction():
    env = create_mj_env(OmegaConf.create({"name": "MetaReacher"}))
    prediction = GroundTruth(OmegaConf.create({"name": "MetaReacher"}))
    state = env.reset()
    next_state = prediction.next_state(state[None], env.sample_random_action()[None])
    assert next_state.shape == state.shape
