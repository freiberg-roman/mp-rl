import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mprl.config import ConfigRepository
from mprl.env import MujocoFactory
from mprl.models.physics.moe import MixtureOfExperts
from mprl.utils.buffer.random_replay_buffer import RandomRB


def get_prep_fn(name: str):
    if name == "HalfCheetah":
        return lambda x: x[..., 1:]
    else:
        raise ValueError(f"Unknown environment {name}")


def fill_buffer(num_steps, env, buffer):
    state, sim_state = env.reset(time_out_after=500)
    for _ in tqdm(range(num_steps)):
        action = env.random_action()
        next_state, reward, done, timeout, sim_state, _ = env.step(action)
        buffer.add(state, next_state, action, reward, done, sim_state)
        if done or timeout:
            state, sim_state = env.reset(time_out_after=500)
        else:
            state = next_state
    return buffer


def update_moe(batch, moe):
    (states, next_states, actions, _, _, _) = batch.to_torch_batch()
    update_loss = moe.update(states, actions, next_states)
    return update_loss


def evaluate_moe(batch, moe):
    (states, next_states, actions, _, _, _) = batch.to_torch_batch()
    pred_next_states = moe.next_state(states, actions)
    return (pred_next_states - next_states).pow(2).mean().item()


@hydra.main(config_path="../config", config_name="main.yaml")
def run(cfg: DictConfig):
    """
    Simple script to train and evaluate a physics model, given a random policy generated data set.
    """

    # Dependency Injection
    config_repository = ConfigRepository(cfg)
    config_env = config_repository.get_environment_config()
    config_model = config_repository.get_model_config()
    # Setup
    training_environment = MujocoFactory(env_config_gateway=config_repository).create()
    buffer = RandomRB(
        capacity=1000000,
        state_dim=config_env.state_dim,
        action_dim=config_env.action_dim,
        sim_qpos_dim=config_env.sim_qpos_dim,
        sim_qvel_dim=config_env.sim_qvel_dim,
    )
    # Fill buffer
    print("Filling buffer")
    buffer_train = fill_buffer(1000000, training_environment, buffer)
    buffer_eval = fill_buffer(1000000, training_environment, buffer)
    model = MixtureOfExperts(
        state_dim_in=config_model.state_dim_in,
        state_dim_out=config_model.state_dim_out,
        action_dim=config_model.action_dim,
        num_experts=config_model.num_experts,
        network_width=config_model.network_width,
        variance=config_model.variance,
        use_batch_normalization=config_model.use_batch_normalization,
    )

    cfg_wandb = cfg.logging
    wandb.init(
        project="physics-models",
        name="model_training_" + str(cfg_wandb.run_id),
        config=OmegaConf.to_container(config_repository.get_config()),
        mode="online",
    )

    # Main loop
    print("Training model")
    for _ in tqdm(range(1000000)):
        train_batch = next(buffer_train.get_iter(it=1, batch_size=128))
        eval_batch = next(buffer_eval.get_iter(it=1, batch_size=128))

        # Train and evaluate model
        update_loss = update_moe(train_batch, model)
        mse_loss_eval = evaluate_moe(eval_batch, model)
        mse_loss_train = evaluate_moe(train_batch, model)
        wandb.log(
            {
                **update_loss,
                "mse_loss_eval": mse_loss_eval,
                "mse_loss_train": mse_loss_train,
            }
        )


if __name__ == "__main__":
    run()
