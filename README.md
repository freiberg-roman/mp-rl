# MP-RL
Pytorch implementation of sequenced based RL algorithms.

### Installation
Just clone the repository and run ``pip`` with the needed requirements. Note that a valid MuJoCo 2.1.0 installation is required.

    git clone https://github.com/freiberg-roman/mp-rl.git
    pip install -e ".[dev]"

### Run experiments
To run experiments, use the following command:
```python
    python -m mprl.main.run algorithm=sac env=half_cheetah
    # or sac_mp, sac_stepwise_mp
```

