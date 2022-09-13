import numpy as np
from mprl.utils.ds_helper import to_ts


class MetaController:

     def get_action(self, desired_pos, desired_vel, current_pos, current_vel):
         des_pos = to_ts(desired_pos)
         c_pos = to_ts(current_pos)
         gripper_pos = des_pos[-1]

         cur_pos = c_pos[:-1]
         xyz_pos = des_pos[:-1]

         trq = np.hstack([(xyz_pos - cur_pos), gripper_pos])
         return trq