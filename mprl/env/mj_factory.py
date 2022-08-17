import pathlib

from omegaconf import DictConfig

from mprl.env.mujoco import (
    AntEnv,
    HalfCheetahEnv,
    HopperEnv,
    HumanoidEnv,
    ReacherEnv,
    SawyerPickPlaceEnvV2,
    SawyerPushEnvV2,
    SawyerReachEnvV2,
)
from mprl.env.mujoco.mj_env import MujocoEnv

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"


def create_mj_env(cfg: DictConfig) -> MujocoEnv:
    if cfg.name == "HalfCheetah":
        return HalfCheetahEnv(base=BASE)
    if cfg.name == "Ant":
        return AntEnv(base=BASE)
    if cfg.name == "Hopper":
        return HopperEnv(base=BASE)
    if cfg.name == "Humanoid":
        return HumanoidEnv(base=BASE)
    if cfg.name == "Reacher":
        return ReacherEnv(base=BASE)
    if cfg.name == "MetaReacher":
        return SawyerReachEnvV2(base=BASE + "meta/")
    if cfg.name == "MetaPush":
        return SawyerPushEnvV2(base=BASE + "meta/")
    if cfg.name == "MetaButtonPress":
        pass
    if cfg.name == "MetaPickAndPlace":
        return SawyerPickPlaceEnvV2(base=BASE + "meta/")
    if cfg.name == "MetaWindowOpen":
        pass
    if cfg.name == "MetaWindowClose":
        pass
