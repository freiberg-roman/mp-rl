import mujoco
import numpy as np

from ..util import hamacher_product, tolerance
from .base_pd import SawyerPD


class MetaPDWindowOpen(SawyerPD):
    TARGET_RADIUS = 0.05

    def __init__(self, base):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)
        self.base = base

        super().__init__(
            base + "meta_pd_window_open.xml",
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_angle": np.array(
                [
                    0.3,
                ],
                dtype=np.float32,
            ),
            "obj_init_pos": np.array([-0.1, 0.785, 0.16], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = (
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = (np.array(goal_low), np.array(goal_high))

        self.maxPullDist = 0.2
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

    def load_assets(self):
        ASSETS = {}

        # files to load
        mesh_files = [
            "base.stl",
            "block.stl",
            "eGripperBase.stl",
            "head.stl",
            "l0.stl",
            "l1.stl",
            "l2.stl",
            "l3.stl",
            "l4.stl",
            "l5.stl",
            "l6.stl",
            "pedestal.stl",
            "tablebody.stl",
            "tabletop.stl",
            "window/window_base.stl",
            "window/window_frame.stl",
            "window/window_h_frame.stl",
            "window/window_h_base.stl",
            "window/windowa_frame.stl",
            "window/windowa_glass.stl",
            "window/windowa_h_frame.stl",
            "window/windowa_h_glass.stl",
            "window/windowb_frame.stl",
            "window/windowb_glass.stl",
            "window/windowb_h_frame.stl",
            "window/windowb_h_glass.stl",
        ]
        texuture_files = [
            "floor2.png",
            "metal.png",
            "wood2.png",
            "wood4.png",
        ]
        xml_files = [
            "basic_scene.xml",
            "window_dependencies.xml",
            "window_horiz.xml",
            "xyz_base_dependencies.xml",
            "xyz_base.xml",
        ]
        for file in mesh_files:
            with open(self.base + "meshes/" + file, "rb") as f:
                ASSETS[file] = f.read()

        for file in texuture_files:
            with open(self.base + "textures/" + file, "rb") as f:
                ASSETS[file] = f.read()

        for file in xml_files:
            with open(self.base + file, "rb") as f:
                ASSETS[file] = f.read()
        return ASSETS

    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self._get_site_pos("handleOpenStart")

    def _get_quat_objects(self):
        return np.zeros(4)

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()

        self._target_pos = self.obj_init_pos + np.array([0.2, 0.0, 0.0])

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "window")
        ] = self.obj_init_pos
        self.window_handle_pos_init = self._get_pos_objects()
        self.set_joint_qpos("window_slide", 0.0)

        return self._get_obs()

    def compute_reward(self, actions, obs):
        del actions
        obj = self._get_pos_objects()
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = obj[0] - target[0]
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = self.obj_init_pos[0] - target[0]
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.window_handle_pos_init - self.init_tcp)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid="long_tail",
        )
        tcp_opened = 0
        object_grasped = reach

        reward = 10 * hamacher_product(reach, in_place)
        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)
