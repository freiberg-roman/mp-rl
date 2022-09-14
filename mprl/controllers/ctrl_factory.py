from omegaconf.omegaconf import DictConfig

from .meta_controller import MetaController
from .pd_controller import PDController


def get_ctrl(env_cfg: DictConfig, ctrl_cfg: DictConfig):
    if env_cfg.name.contains("metaworld"):
        return MetaController()
    else:
        return PDController(ctrl_cfg)
