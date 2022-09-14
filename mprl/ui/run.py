import hydra
from omegaconf import DictConfig

from mprl.config import ConfigRepository


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):
    config_repository = ConfigRepository(cfg)

    if cfg.algorithm.name == "sac":
        train_sac(cfg.algorithm, cfg.env, cfg.logger)

    if cfg.algorithm.name == "mp_sac":
        train_mp_sac_vanilla(
            cfg.algorithm, cfg.env, cfg.logger
        )  # inefficient version as reference

    if cfg.algorithm.name == "sac_mp_stepwise":
        if cfg.algorithm.prediction.name == "OffPolicy":
            train_stepwise_mp_sac_offpolicy(cfg.algorithm, cfg.env, cfg.logger)
        elif (
            cfg.algorithm.prediction.name == "GroundTruth"
            or cfg.algorithm.prediction.name == "MixtureOfExperts"
        ):
            train_stepwise_mp_sac(cfg.algorithm, cfg.env, cfg.logger)
        else:
            raise ValueError("Unknown prediction type: cfg.algorithm.prediction")


if __name__ == "__main__":
    run()
