from omegaconf import DictConfig

from mprl.env import MujocoFactory
from mprl.pipeline import Trainer


class StubConfig:
    def get_training_config(self):
        return DictConfig(
            {
                "steps_per_epoch": 10,
                "total_steps": 100,
                "warm_start_steps": 1,
                "update_after_first": 1,
                "update_each": 1,
                "update_for": 1,
                "time_out_after": 1,
            }
        )


def test_load_store_trainer():
    test_env = MujocoFactory(None).get_test_env()
    test_env_two = MujocoFactory(None).get_test_env()
    test_env_two._total_steps = 1

    trainer = Trainer(test_env, train_config_gateway=StubConfig())
    trainer_two = Trainer(test_env, train_config_gateway=StubConfig())
    trainer.store(trainer.store_under("./"))
    trainer_two.load(trainer.store_under("./"))
    assert trainer_two.env.total_steps == 0

    import shutil

    shutil.rmtree(trainer.store_under("./"))
