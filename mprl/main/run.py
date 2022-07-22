import hydra
from omegaconf import DictConfig

from mprl.main.routines import (
    train_mp_sac_vanilla,
    train_sac,
    train_stepwise_mp_sac,
    train_stepwise_mp_sac_offpolicy,
)


@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):

    if cfg.algorithm.name == "sac":
        train_sac(cfg.algorithm, cfg.env, cfg.logger)

    if cfg.algorithm.name == "mp_sac":
        train_mp_sac_vanilla(
            cfg.algorithm, cfg.env, cfg.logger
        )  # inefficient version as reference

    if cfg.algorithm.name == "sac_mp_stepwise":
        if cfg.algorithm.prediction == "off_policy":
            train_stepwise_mp_sac_offpolicy(cfg.algorithm, cfg.env, cfg.logger)
        elif cfg.algorithm.prediction == "ground_truth":
            train_stepwise_mp_sac(cfg.algorithm, cfg.env, cfg.logger)
        elif cfg.algorithm.prediction == "moe":
            pass  # TODO
        else:
            raise ValueError("Unknown prediction type: cfg.algorithm.prediction")


if __name__ == "__main__":
    run()
