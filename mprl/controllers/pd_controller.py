import numpy as np
import torch

from mprl.utils.ds_helper import to_np, to_ts


class PDController:
    def __init__(self, cfg):
        self.pgains = torch.tensor(cfg.pgains)
        self.dgains = torch.tensor(cfg.dgains)

    def get_action(self, desired_pos, desired_vel, current_pos, current_vel, bias=None):
        desired_pos = to_ts(desired_pos)
        desired_vel = to_ts(desired_vel)
        current_pos = to_ts(current_pos)
        current_vel = to_ts(current_vel)
        if bias is not None:
            bias = to_np(bias)
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgains * qd_d + self.dgains * vd_d
        if bias is not None:
            return torch.clamp(
                target_j_acc + bias, -1, 1
            )  # in most cases additional forces

        return torch.clamp(target_j_acc, -1, 1)
