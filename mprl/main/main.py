import hydra
from omegaconf import DictConfig

from mprl.main.routines import (
    train_mp_sac_augmented,
    train_mp_sac_vanilla,
    train_mp_sac_virtual,
    train_sac,
    train_stepwise_mp_sac,
)


@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):

    if cfg.mode == "train_sac":
        train_sac(cfg)

    if cfg.mode == "train_mp_sac":
        if cfg.buffer.mode == "sequences":
            train_mp_sac_vanilla(cfg)

        elif cfg.buffer.mode == "virtual":
            train_mp_sac_virtual(cfg)

        elif cfg.buffer.mode == "augmented":
            train_mp_sac_augmented(cfg)

    if cfg.mode == "train_stepwise_mp_sac":
        train_stepwise_mp_sac(cfg)


if __name__ == "__main__":
    run()
