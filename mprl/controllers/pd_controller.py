import numpy as np


class PDController:
    def __init__(self, cfg):
        self.pgains = np.array(cfg.pgains)
        self.dgains = np.array(cfg.dgains)

    def ctrl(self, desired_pos, desired_vel, current_pos, current_vel, bias=None):
        desired_pos = desired_pos.cpu().detach().numpy()
        desired_vel = desired_vel.cpu().detach().numpy()
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgains * qd_d + self.dgains * vd_d
        if bias is not None:
            return target_j_acc + bias  # in most cases gravity forces

        return target_j_acc
