from typing import Optional

import mujoco
import numpy as np
from gym.spaces import Box

from mprl.env.mujoco.meta.util import hamacher_product, tolerance
from mprl.env.mujoco.mj_env import MujocoEnv

from .base_sawyer import BaseSawyer


class SawyerWindowOpenEnvV2(BaseSawyer):
    """
    Motivation for V2:
        When V1 scripted policy failed, it was often due to limited path length.
    Changelog from V1 to V2:
        - (8/11/20) Updated to Byron's XML
        - (7/7/20) Added 3 element handle position to the observation
            (for consistency with other environments)
        - (6/15/20) Increased max_path_length from 150 to 200
    """

    TARGET_RADIUS = 0.05

    def __init__(self, base):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)

        model_name = "sawyer_window_horizontal.xml"
        frame_skip = 5
        mocap_low = None
        mocap_high = None
        action_scale = 1.0 / 100
        action_rot_scale = 1.0
        self.base = base
        assets = self.load_assets()

        MujocoEnv.__init__(
            self, base + model_name, frame_skip=frame_skip, assets=assets
        )
        self.reset_mocap_welds()
        self.random_init = True
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length = 0
        self.seeded_rand_vec = False
        self._freeze_rand_vec = True
        self._last_rand_vec = None

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self.isV2 = "V2" in type(self).__name__
        # Technically these observation lengths are different between v1 and v2,
        # but we handle that elsewhere and just stick with v2 numbers here
        self._obs_obj_max_len = 14 if self.isV2 else 6
        self._obs_obj_possible_lens = (6, 14)

        self._set_task_called = False
        self._partially_observable = True

        self.hand_init_pos = None  # OVERRIDE ME
        self._target_pos = None  # OVERRIDE ME
        self._random_reset_space = None  # OVERRIDE ME

        self._last_stable_obs = None
        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of _prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)
        self._prev_obs = self._get_curr_obs_combined_no_goal()

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

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

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

        return reward

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

    def compute_reward(self, obs, actions):
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

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        mujoco.mj_forward(self.model, self.data)

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)'
            {pos, quat} into a single flat observation. The goal's position is
            *not* included in this.

        Returns:
            np.ndarray: The flat observation array (18 elements)

        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        assert len(obs_obj_padded) in self._obs_obj_possible_lens
        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))

    def _get_obs(self):
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the
            goal position to form a single flat observation.

        Returns:
            np.ndarray: The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def reset(self, time_out_after: Optional[int] = None):
        self.curr_path_length = 0
        return super().reset(time_out_after=time_out_after)

    def _get_state_rand_vec(self):
        rand_vec = np.random.uniform(
            self._random_reset_space.low,
            self._random_reset_space.high,
            size=self._random_reset_space.low.size,
        )
        self._last_rand_vec = rand_vec
        return rand_vec

    def get_endeff_pos(self):
        return self.data.body("hand").xpos

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self._get_site_pos("rightEndEffector")
        left_finger_pos = self._get_site_pos("leftEndEffector")
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos[:] = new_mocap_pos
        self.data.mocap_quat[:] = np.array([[1, 0, 1, 0]])

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, siteName):
        return self.model.site(siteName).pos.copy()

    def _set_pos_site(self, name, pos):
        """Sets the position of the site corresponding to `name`

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1
        self.data.site(name).xpos[:] = pos[:3]

    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def set_joint_qpos(self, name, value):

        addr = self.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.data.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (
                end_i - start_i,
            ), "Value has incorrect shape %s: %s" % (name, value)
            self.data.qpos[start_i:end_i] = value

    def get_joint_qpos_addr(self, name):
        """
        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_type = self.model.jnt_type[joint_id]
        joint_addr = self.model.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            )
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets

        :rtype: list of (str, np.ndarray)
        """
        return [("goal", self._target_pos)]
