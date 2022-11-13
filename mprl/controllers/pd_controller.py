from typing import Union

import numpy as np
import torch as ch

from mprl.utils.ds_helper import to_ts

from .ctrl import Controller


class PDController(Controller):
    def __init__(
        self,
        pgains: Union[np.ndarray, ch.Tensor],
        dgains: Union[np.ndarray, ch.Tensor],
    ):
        self.pgains: ch.Tensor = to_ts(pgains)
        self.dgains: ch.Tensor = to_ts(dgains)

    def get_action(
        self,
        desired_pos: Union[np.ndarray, ch.Tensor],
        desired_vel: Union[np.ndarray, ch.Tensor],
        current_pos: Union[np.ndarray, ch.Tensor],
        current_vel: Union[np.ndarray, ch.Tensor],
        action_clip: bool = False,
    ):
        desired_pos = to_ts(desired_pos)
        desired_vel = to_ts(desired_vel)
        current_pos = to_ts(current_pos)
        current_vel = to_ts(current_vel)
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgains * qd_d + self.dgains * vd_d
        if action_clip:
            target_j_acc = target_j_acc.clamp(-1, 1)
        return target_j_acc
