import torch

from mprl.utils.ds_helper import to_ts

from .ctrl import Controller


class MetaController(Controller):
    def __init__(self, pgains: torch.Tensor):
        self.pgains = to_ts(pgains)  # 3 dim for xyz

    def get_action(self, desired_pos, desired_vel, current_pos, current_vel):
        _, _ = desired_vel, current_vel  # not used
        des_pos = to_ts(desired_pos)
        c_pos = to_ts(current_pos)
        gripper_pos = to_ts(des_pos[..., [-1]])

        cur_pos = c_pos[..., :-1]
        xyz_pos = des_pos[..., :-1]

        xyz_pos = torch.squeeze(xyz_pos)
        cur_pos = torch.squeeze(cur_pos)

        trq = torch.hstack([self.pgains * (xyz_pos - cur_pos), gripper_pos])
        return torch.clamp(trq, -1, 1)
