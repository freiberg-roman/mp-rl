import numpy as np
import wandb

if __name__ == "__main__":
    wandb.init(
        project="test",
        name="simple_run",
    )

    for _ in range(100):
        wandb.log(
            {
                "hist": wandb.Histogram(
                    np_histogram=np.histogram(np.random.uniform(size=(100,)))
                )
            }
        )
