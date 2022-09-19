import hydra
from omegaconf import DictConfig

from mprl.di import Container
from mprl.models import SACFactory
from mprl.pipeline import Evaluator, Trainer

from .routine import train_sac


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):

    # Dependency Injection
    container = Container()
    container.config.hydra_configuration = cfg
    container.wire(modules=[__name__])

    if cfg.alg.name == "sac":
        agent = SACFactory().create()
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
    training_environment = None
    evaluation_environment = None

    trainer = Trainer(training_environment)
    evaluator = Evaluator(evaluation_environment)

    while trainer.training_steps_left:
        agent = trainer.train(agent)
        evaluator.evaluate(agent)


if __name__ == "__main__":
    run()
