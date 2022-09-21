from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig

from mprl.utils.ds_helper import to_ts


class PDController:
    def __init__(
        self,
        pgains: Union[np.ndarray, torch.Tensor],
        dgains: Union[np.ndarray, torch.Tensor],
        device: torch.device,
    ):
        self.pgains: torch.Tensor = to_ts(pgains).to(device)
        self.dgains: torch.Tensor = to_ts(dgains).to(device)

    def get_action(
        self,
        desired_pos: Union[np.ndarray, torch.Tensor],
        desired_vel: Union[np.ndarray, torch.Tensor],
        current_pos: Union[np.ndarray, torch.Tensor],
        current_vel: Union[np.ndarray, torch.Tensor],
    ):
        desired_pos = to_ts(desired_pos)
        desired_vel = to_ts(desired_vel)
        current_pos = to_ts(current_pos)
        current_vel = to_ts(current_vel)
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgains * qd_d + self.dgains * vd_d
        return torch.clamp(target_j_acc, -1, 1)
