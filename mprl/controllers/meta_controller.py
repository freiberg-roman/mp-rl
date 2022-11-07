import torch as ch

from mprl.utils.ds_helper import to_ts

from .ctrl import Controller


class MetaController(Controller):
    """Simple position controller."""

    def __init__(self, pgains: ch.Tensor):
        self.pgains: ch.Tensor = to_ts(
            pgains
        )  # 3 dim for xyz in task space and normalized position of gripper

    def get_action(self, desired_pos, desired_vel, current_pos, current_vel):
        _, _ = desired_vel, current_vel  # not used
        des_pos = to_ts(desired_pos)
        c_pos = to_ts(current_pos)
        trq = self.pgains * (des_pos - c_pos)
        return trq
