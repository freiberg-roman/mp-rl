import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from mprl.config import ConfigRepository
from mprl.env import MujocoFactory
from mprl.models import SACFactory, SACMixedMPFactory
from mprl.pipeline import Evaluator, Trainer


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):

    # Dependency Injection
    config_repository = ConfigRepository(cfg)

    if cfg.alg.name == "sac":
        agent = SACFactory(config_gateway=config_repository).create()
    elif cfg.alg.name == "sac_mixed_mp":
        agent = SACMixedMPFactory(
            config_gateway=config_repository, env_config_gateway=config_repository
        ).create()
    elif cfg.alg.name == "sac_mp":
        ...  # TODO
    else:
        raise ValueError(f"Unknown algorithm {cfg.alg.name}")

    # Setup
    training_environment = MujocoFactory(env_config_gateway=config_repository).create()
    evaluation_environment = MujocoFactory(
        env_config_gateway=config_repository
    ).create()

    trainer = Trainer(training_environment, train_config_gateway=config_repository)
    evaluator = Evaluator(evaluation_environment, eval_config_gateway=config_repository)

    cfg_wandb = cfg.logging
    wandb.init(
        project=cfg_wandb.project,
        name=cfg_wandb.name + "_" + str(cfg_wandb.run_id),
        config=OmegaConf.to_container(config_repository.get_config()),
        mode=cfg_wandb.mode,
    )

    # Main loop
    while trainer.has_training_steps_left:
        result = evaluator.evaluate(
            agent, after_performed_steps=trainer.performed_training_steps
        )
        print(result)
        agent = trainer.train_one_epoch(agent)


if __name__ == "__main__":
    run()
