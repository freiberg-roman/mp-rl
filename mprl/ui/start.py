import os

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from mprl.config import ConfigRepository
from mprl.env import MujocoFactory
from mprl.pipeline import Evaluator, Trainer
from mprl.pipeline.checkpoint import CheckPoint


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):

    # Dependency Injection
    config_repository = ConfigRepository(cfg)

    if cfg.alg.name == "sac":
        from mprl.models import SACFactory

        agent = SACFactory(config_gateway=config_repository).create()
    elif cfg.alg.name == "sac_mixed_mp":
        from mprl.models import SACMixedMPFactory

        agent = SACMixedMPFactory(
            config_gateway=config_repository, env_config_gateway=config_repository
        ).create()
    elif cfg.alg.name == "sac_mp":
        from mprl.models import SACMPFactory

        agent = SACMPFactory(
            config_gateway=config_repository, env_config_gateway=config_repository
        ).create()
    elif cfg.alg.name == "sac_tr":
        from mprl.models import SACTRFactory

        agent = SACTRFactory(
            config_gateway=config_repository, env_config_gateway=config_repository
        ).create()
    else:
        raise ValueError(f"Unknown algorithm {cfg.alg.name}")

    # Setup
    training_environment = MujocoFactory(env_config_gateway=config_repository).create()
    evaluation_environment = MujocoFactory(
        env_config_gateway=config_repository
    ).create()

    trainer = Trainer(training_environment, train_config_gateway=config_repository)
    evaluator = Evaluator(evaluation_environment, eval_config_gateway=config_repository)

    # Checkpointing
    dir_path = os.getcwd()
    checkpoint = CheckPoint([trainer, agent], dir_path)
    if cfg.continue_run or cfg.eval_current:
        checkpoint.restore_checkpoint(cfg.checkpoint_source, cfg.restore_steps_after)

    # Logging
    cfg_wandb = cfg.logging
    wandb.init(
        project=cfg_wandb.project,
        name=cfg_wandb.name + "_" + str(cfg_wandb.run_id),
        config=OmegaConf.to_container(config_repository.get_config()),
        mode=cfg_wandb.mode,
    )

    if not cfg.eval_current:
        # Main loop
        checkpoint.create_checkpoint(trainer.performed_training_steps)
        while trainer.has_training_steps_left:
            result = evaluator.evaluate(
                agent, after_performed_steps=trainer.performed_training_steps
            )
            print(result)
            wandb.log(result)
            agent = trainer.train_one_epoch(agent)

            if trainer.performed_training_steps % cfg.train.checkpoint_each == 0:
                checkpoint.create_checkpoint(trainer.performed_training_steps)
    else:
        result = evaluator.evaluate(
            agent, after_performed_steps=trainer.performed_training_steps, render=True
        )
        print(result)
        wandb.log(result)


if __name__ == "__main__":
    run()
