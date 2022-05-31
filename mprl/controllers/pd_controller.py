class PDController:
    def __init__(self, cfg):
        self.pgian = cfg.pgains
        self.dgian = cfg.dgains

    def ctrl(self, desired_pos, desired_vel, current_pos, current_vel, bias=None):
        qd_d = desired_pos - current_pos
        vd_d = desired_vel - current_vel
        target_j_acc = self.pgain * qd_d + self.dgain * vd_d
        if bias is not None:
            return target_j_acc + bias  # in most cases gravity forces

        return target_j_acc
