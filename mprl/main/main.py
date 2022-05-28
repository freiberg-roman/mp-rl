import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from mprl.env import create_mj_env

@hydra.main(config_path="configs", config_name="main.yaml")
def run(cfg: DictConfig):
    env = create_mj_env("HalfCheetah")

    env.reset()

    for i in range(1000):
        act = np.random.uniform(low=-1., high=1., size=(6,))
        env.step(act)
        env.render(mode="human")


if __name__ == "__main__":
    run()