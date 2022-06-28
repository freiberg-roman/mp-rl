import hydra
from omegaconf import DictConfig

from mprl.main.routines import train_mp_sac_vanilla, train_sac, train_stepwise_mp_sac


@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):

    if cfg.algorithm.mode == "sac":
        train_sac(cfg.algorithm, cfg.env, cfg.logger)

    if cfg.algorithm.name == "mp_sac":
        train_mp_sac_vanilla(
            cfg.algorithm, cfg.env, cfg.logger
        )  # inefficient version as reference

    if cfg.mode == "sac_mp_stepwise":
        train_stepwise_mp_sac(cfg.algorithm, cfg.env, cfg.logger)


if __name__ == "__main__":
    run()
