import numpy as np
import wandb

if __name__ == "__main__":
    id = wandb.util.generate_id()
    print(id)
    wandb.init(
        project="test-wandb",
        name="simple_run",
        id="1y24vfao",
        resume="must",
    )

    for i in range(0, 400):
        wandb.log(
            {
                "value": (i**2) / 100 + 10,
            },
            step=i,
        )
