import pathlib

from .config_gateway import EnvConfigGateway
from .mp_rl_environment import MPRLEnvironment
from .mujoco import (
    AntEnv,
    HalfCheetahEnv,
    HopperEnv,
    MetaPDButtonPress,
    MetaPDReacher,
    MetaPDWindowOpen,
    MetaPosButtonPress,
    MetaPosReacher,
    MetaPosWindowOpen,
    ReacherEnv,
)

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"


class MujocoFactory:
    def __init__(self, env_config_gateway: EnvConfigGateway):
        self.cfg = env_config_gateway

    def create(self) -> MPRLEnvironment:
        cfg = self.cfg

        if cfg.get_env_name() == "HalfCheetah":
            return HalfCheetahEnv(base=BASE)
        if cfg.get_env_name() == "Ant":
            return AntEnv(base=BASE)
        if cfg.get_env_name() == "Hopper":
            return HopperEnv(base=BASE)
        if cfg.get_env_name() == "Reacher":
            return ReacherEnv(base=BASE)
        if cfg.get_env_name() == "MetaPosReacher":
            return MetaPosReacher(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPosWindowOpen":
            return MetaPosWindowOpen(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPosButtonPress":
            return MetaPosButtonPress(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPDReacher":
            return MetaPDReacher(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPDWindowOpen":
            return MetaPDWindowOpen(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPDButtonPress":
            return MetaPDButtonPress(base=BASE + "meta/")
