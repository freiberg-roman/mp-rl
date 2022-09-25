import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..util import hamacher_product, tolerance
from .base_pd import SawyerPD


class MetaPDButtonPress(SawyerPD):
    def __init__(self, base):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.85, 0.115)
        obj_high = (0.1, 0.9, 0.115)
        self.base = base

        super().__init__(
            base + "meta_pd_button_press.xml",
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_pos": np.array([0.0, 0.9, 0.115], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.78, 0.12])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = (
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = (np.array(goal_low), np.array(goal_high))

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
            "button/button.stl",
            "button/buttonring.stl",
            "button/stopbot.stl",
            "button/stopbutton.stl",
            "button/stopbuttonrim.stl",
            "button/stopbuttonrod.stl",
            "button/stoptop.stl",
        ]
        texuture_files = [
            "floor2.png",
            "metal.png",
            "wood2.png",
            "wood4.png",
            "button/metal1.png",
        ]
        xml_files = [
            "basic_scene.xml",
            "buttonbox_dependencies.xml",
            "buttonbox.xml",
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
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.compute_reward(action, obs)

        info = {
            "success": float(obj_to_target <= 0.02),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": float(tcp_open > 0),
            "grasp_reward": near_button,
            "in_place_reward": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("btnGeom")

    def _get_pos_objects(self):
        return self.get_body_com("button") + np.array([0.0, -0.193, 0.0])

    def _get_quat_objects(self):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button")
        mat = self.data.geom_xmat[id].reshape((3, 3))
        return Rotation.from_matrix(mat).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_sim_state((qpos, qvel))

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos

        self.model.body_pos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        ] = self.obj_init_pos
        self._set_obj_xyz(0)
        self._target_pos = self._get_site_pos("hole")

        self._obj_to_target_init = abs(
            self._target_pos[1] - self._get_site_pos("buttonStart")[1]
        )

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid="long_tail",
        )
        button_pressed = tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid="long_tail",
        )

        reward = 2 * hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)

    @property
    def name(self) -> str:
        return "MetaPDButtonPress"
