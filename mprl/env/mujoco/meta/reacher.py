import os
import pickle
from typing import Optional

import mujoco
import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from mprl.env.mujoco.meta.util import tolerance
from mprl.env.mujoco.mj_env import MujocoEnv


class SawyerReachEnvV2(MujocoEnv):
    """
    Adapted SawyerReachEnvV2 environment for new MuJoCo bindings
    """

    def __init__(self, base):
        goal_low = (-0.1, 0.8, 0.05)
        goal_high = (0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        action_scale = (1.0 / 100,)
        action_rot_scale = (1.0,)
        frame_skip = 5
        model_name = "sawyer_reach_v2.xml"
        self.base = base
        assets = self.load_assets()

        MujocoEnv.__init__(
            self, base + model_name, frame_skip=frame_skip, assets=assets
        )
        self.reset_mocap_welds()

        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.mocap_low = np.hstack(hand_low)
        self.mocap_high = np.hstack(hand_high)
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
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.6, 0.02]),
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

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
        ]
        texuture_files = [
            "floor2.png",
            "metal.png",
            "wood2.png",
            "wood4.png",
        ]
        xml_files = [
            "basic_scene.xml",
            "block_dependencies.xml",
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

    @property
    def model_name(self):
        return os.path.join("sawyer_xyz/sawyer_reach_v2.xml")

    def evaluate_state(self, obs, action):

        reward, reach_dist, in_place = self.compute_reward(action, obs)
        success = float(reach_dist <= 0.05)

        info = {
            "success": success,
            "near_object": reach_dist,
            "grasp_success": 1.0,
            "grasp_reward": reach_dist,
            "in_place_reward": in_place,
            "obj_to_target": reach_dist,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self.get_body_com("obj")

    def _get_quat_objects(self):
        id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        mat = self.data.geom_xmat[id].reshape((3, 3))
        return Rotation.from_matrix(mat).as_quat()

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com("obj")[:2] - self.get_body_com("obj")[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [adjusted_pos[0], adjusted_pos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1

        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        target = self._target_pos

        tcp_to_target = np.linalg.norm(tcp - target)

        in_place_margin = np.linalg.norm(self.hand_init_pos - target)
        in_place = tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        return [10 * in_place, tcp_to_target, in_place]

    _HAND_SPACE = Box(
        np.array([-0.525, 0.348, -0.0525]), np.array([+0.525, 1.025, 0.7])
    )
    max_path_length = 500

    TARGET_RADIUS = 0.05

    def set_task(self, task):
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._last_rand_vec = data["rand_vec"]
        self._freeze_rand_vec = True
        self._last_rand_vec = data["rand_vec"]
        del data["rand_vec"]
        self._partially_observable = data["partially_observable"]
        del data["partially_observable"]
        self.reset()

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

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets

        :rtype: list of (str, np.ndarray)
        """
        return [("goal", self._target_pos)]

    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

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

    def _get_obs_dict(self):
        obs = self._get_obs()
        return {
            "state_observation": obs,
            "state_desired_goal": self._get_pos_goal(),
            "state_achieved_goal": obs[3:-3],
        }

    @property
    def observation_space(self):
        obs_obj_max_len = self._obs_obj_max_len if self.isV2 else 6

        obj_low = np.full(obs_obj_max_len, -np.inf)
        obj_high = np.full(obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0

        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
        )

    def sample_random_action(self):
        return np.random.uniform(-1, 1, (4,))

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]], self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        self.current_steps += 1
        self._total_steps += 1
        done = (
            self.time_out_after is not None
            and self.current_steps >= self.time_out_after
        )
        self._last_stable_obs = self._get_obs()
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        return self._last_stable_obs, reward, False, done

    def reset(self, time_out_after: Optional[int] = None):
        self.curr_path_length = 0
        return super().reset(time_out_after=time_out_after)

    def _reset_hand(self, steps=50):
        for _ in range(steps):
            self.data.mocap_pos[0, :] = self.hand_init_pos
            self.data.mocap_quat[0, :] = np.array([[1, 0, 1, 0]])
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

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

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i, :] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        mujoco.mj_forward(self.model, self.data)

    @property
    def total_steps(self):
        return self._total_steps
