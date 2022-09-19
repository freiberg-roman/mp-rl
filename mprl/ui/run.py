import hydra
from omegaconf import DictConfig

from mprl.config import ConfigRepository
from mprl.env import MujocoFactory
from mprl.models import SACFactory
from mprl.pipeline import Evaluator, Trainer


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):

    # Dependency Injection
    config_repository = ConfigRepository(cfg)

    if cfg.alg.name == "sac":
        agent = SACFactory(config_gateway=config_repository).create()
    elif cfg.alg.name == "sac_mp_stepwise":
        ...  # TODO
    elif cfg.alg.name == "sac_mp":
        ...  # TODO
    else:
        raise ValueError(f"Unknown algorithm {cfg.alg.name}")

    # if cfg.algorithm.name == "sac":
    #     train_sac(cfg.algorithm, cfg.env, cfg.logger)

    # if cfg.algorithm.name == "mp_sac":
    #     train_mp_sac_vanilla(
    #         cfg.algorithm, cfg.env, cfg.logger
    #     )  # inefficient version as reference

    # if cfg.algorithm.name == "sac_mp_stepwise":
    #     if cfg.algorithm.prediction.name == "OffPolicy":
    #         train_stepwise_mp_sac_offpolicy(cfg.algorithm, cfg.env, cfg.logger)
    #     elif (
    #         cfg.algorithm.prediction.name == "GroundTruth"
    #         or cfg.algorithm.prediction.name == "MixtureOfExperts"
    #     ):
    #         train_stepwise_mp_sac(cfg.algorithm, cfg.env, cfg.logger)
    #     else:
    #         raise ValueError("Unknown prediction type: cfg.algorithm.prediction")

    # Main loop
    training_environment = MujocoFactory(env_config_gateway=config_repository).create()
    evaluation_environment = MujocoFactory(
        env_config_gateway=config_repository
    ).create()

    trainer = Trainer(training_environment, train_config_gateway=config_repository)
    evaluator = Evaluator(evaluation_environment, eval_config_gateway=config_repository)

    while trainer.training_steps_left:
        agent = trainer.train(agent)
        evaluator.evaluate(agent)


if __name__ == "__main__":
    run()
