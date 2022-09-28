This library is under active development! Benchmarks will be added soon.

# Motion Primitive - Reinforcement Learning (MP-RL)
Pytorch implementation of sequenced based RL algorithms. This library contains the following algorithms:
* Soft Actor-Critic (SAC)
* Mition Primitive SAC (MP-SAC)  [*currently under revision*]
* Mixed Motion Primitive SAC (MMP-SAC)

Furthermore, the library contains the following ``gym`` and ``meta_world`` environments:
* HalfCheetah
* Ant
* Hopper
* Reacher
* MetaWorld Positional Reacher
* MetaWorld PD controlled Reacher [*minor modification for PD controller*]
* MetaWorld Positional WindowOpen
* MetaWorld PD controlled WindowOpen [*minor modification for PD controller*]
* MetaWorld Positional ButtonPress
* MetaWorld PD controlled ButtonPress [*minor modification for PD controller*]

In addition you can find the following prediction models:
* Ground Truth [*prediction done by using MuJoCo Simulator*]
* Off-Policy [*prediction done by using replay buffer*]
* Mixure of Experts [*prediction done by using a mixture of experts model -- currently under revision*]

### Installation
Prior to installation, please make sure you have ``MP_Pytorch`` as a dependency installed.
This package is provided in the release note. Please install it first using the following commands:

    cd MP_Pytorch
    pip install -e .

Anything further than personal use of the previously mentioned package requires a license.
Please contact the author for more information.

Also note that a valid MuJoCo 2.1.0 installation is required.
Afterwards, just clone the repository and run ``pip`` with the needed requirements:

    git clone https://github.com/freiberg-roman/mp-rl.git
    pip install -e ".[dev]"

### Run experiments
To run experiments, use the following command:

    python -m mprl.ui.run algorithm={sac, sac_mixed_mp, ..} env={half_cheetah, meta_pos_reacher, ..}
For more settings, please refer to the ``mprl/config`` folder or use the following command:

    python -m mprl.ui.run --help

