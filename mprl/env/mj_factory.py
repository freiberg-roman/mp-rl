import pathlib

from mprl.env.mujoco.half_cheetah import HalfCheetahEnv
from mprl.env.mujoco.mj_env import MujocoEnv

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"


def create_mj_env(name) -> MujocoEnv:
    if name == "HalfCheetah":
        return HalfCheetahEnv(base=BASE)
