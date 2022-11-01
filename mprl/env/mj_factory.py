import pathlib

from .config_gateway import EnvConfigGateway
from .mp_rl_environment import MPRLEnvironment
from .mujoco import AntEnv, HalfCheetahEnv, HopperEnv, ReacherEnv

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
            from .mujoco.meta import OriginalMetaWorld

            return OriginalMetaWorld("reach-v2")
        if cfg.get_env_name() == "MetaPosWindowOpen":
            from .mujoco.meta import OriginalMetaWorld

            return OriginalMetaWorld("window-open-v2")
        if cfg.get_env_name() == "MetaPosButtonPress":
            from .mujoco.meta import OriginalMetaWorld

            return OriginalMetaWorld("button-press-v2")

    @staticmethod
    def get_test_env():
        """Quick access for testing implementations"""
        return HalfCheetahEnv(base=BASE)
