import numpy as np

from mprl.utils.ds_helper import to_np


class PDController:
    def __init__(self, cfg):
        self.pgains = np.array(cfg.pgains)
        self.dgains = np.array(cfg.dgains)

    def get_action(self, desired_pos, desired_vel, current_pos, current_vel, bias=None):
        desired_pos = to_np(desired_pos)
        desired_vel = to_np(desired_vel)
        current_pos = to_np(current_pos)
        current_vel = to_np(current_vel)
        if bias is not None:
            bias = to_np(bias)
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgains * qd_d + self.dgains * vd_d
        if bias is not None:
            return np.clip(
                target_j_acc + bias, -1, 1
            )  # in most cases additional forces

        return np.clip(target_j_acc, -1, 1)
