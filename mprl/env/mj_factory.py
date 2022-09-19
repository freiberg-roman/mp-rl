import pathlib

from mprl.env.mujoco import (
    AntEnv,
    HalfCheetahEnv,
    HopperEnv,
    HumanoidEnv,
    ReacherEnv,
    SawyerButtonPressEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPushEnvV2,
    SawyerReachEnvV2,
    SawyerWindowOpenEnvV2,
)
from mprl.env.mujoco.mj_env import MujocoEnv

from .config_gateway import EnvConfigGateway

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"


class MujocoFactory:
    def __init__(self, env_config_gateway: EnvConfigGateway):
        self.cfg = env_config_gateway

    def create(self) -> MujocoEnv:
        cfg = self.cfg

        if cfg.get_env_name() == "HalfCheetah":
            return HalfCheetahEnv(base=BASE)
        if cfg.get_env_name() == "Ant":
            return AntEnv(base=BASE)
        if cfg.get_env_name() == "Hopper":
            return HopperEnv(base=BASE)
        if cfg.get_env_name() == "Humanoid":
            return HumanoidEnv(base=BASE)
        if cfg.get_env_name() == "Reacher":
            return ReacherEnv(base=BASE)
        if cfg.get_env_name() == "MetaReacher":
            return SawyerReachEnvV2(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPush":
            return SawyerPushEnvV2(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaButtonPress":
            return SawyerButtonPressEnvV2(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaPickAndPlace":
            return SawyerPickPlaceEnvV2(base=BASE + "meta/")
        if cfg.get_env_name() == "MetaWindowOpen":
            return SawyerWindowOpenEnvV2(base=BASE + "meta/")
