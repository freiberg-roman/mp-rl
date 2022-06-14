import pathlib

from mprl.env.mujoco import AntEnv, HalfCheetahEnv, HopperEnv, HumanoidEnv, Reacher
from mprl.env.mujoco.mj_env import MujocoEnv

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"


def create_mj_env(cfg) -> MujocoEnv:
    if cfg.name == "HalfCheetah":
        return HalfCheetahEnv(base=BASE)
    if cfg.name == "Ant":
        return AntEnv(base=BASE)
    if cfg.name == "Hopper":
        return HopperEnv(base=BASE)
    if cfg.name == "Humanoid":
        return HumanoidEnv(base=BASE)
    if cfg.name == "Reacher":
        return Reacher(base=BASE)
