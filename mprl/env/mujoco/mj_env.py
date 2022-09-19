from typing import Optional, Tuple

import mujoco
import numpy as np

from ...utils.ds_helper import to_np
from .rendering.mj_rendering import RenderContextOffscreen, Viewer

DEFAULT_SIZE = 480


class MujocoEnv:
    def __init__(self, path, frame_skip, assets=None):

        if assets is None:
            assets = {}
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(path, assets=assets)
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.init_qpos: np.ndarray = self.data.qpos.ravel().copy()
        self.init_qvel: np.ndarray = self.data.qvel.ravel().copy()
        self._viewers: dict = {}
        self.time_out_after: int = None
        self.current_steps: int = 0
        self._total_steps: int = 0

        self.frame_skip: int = frame_skip

        self.viewer: Viewer = None

        self.metadata: dict = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.low, self.high = self._set_action_space()
        self.jnt_names = self.get_jnt_names
        self.qpos_idx = self.get_qpos_idx(self.jnt_names)
        self.qvel_idx = self.get_qvel_idx(self.jnt_names)
        self.ctrl_idx = self.get_ctrl_idx(self.jnt_names)
        pass

    def _set_action_space(self) -> Tuple[float, float]:
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return low, high

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

    def reset(self, time_out_after: Optional[int] = None) -> np.ndarray:
        self.time_out_after = time_out_after
        self.current_steps = 0
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob, self.get_sim_state()

    def sample_random_action(self):
        """
        Uniform sampling in environment space.
        """

    def set_state(self, qpos, qvel):
        qpos = to_np(qpos)
        qvel = to_np(qvel)
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[self.ctrl_idx] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self._viewers = {}

    def _get_viewer(self, mode, width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = Viewer(self.model, self.data)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = RenderContextOffscreen(
                    width, height, self.model, self.data
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.body(body_name).xpos

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def get_forces(self):
        return self.data.qfrc_bias.flat

    def decompose(self, state):
        if isinstance(state, tuple):
            q, v = state
            return q[..., self.qpos_idx], v[..., self.qvel_idx]

    @property
    def reset_after(self):
        return self.time_out_after

    def get_sim_state(self):
        return self.data.qpos.ravel().copy(), self.data.qvel.ravel().copy()

    def get_qpos_idx(self, names):
        return [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in names
        ]

    def get_qvel_idx(self, names):
        return [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in names
        ]

    def get_ctrl_idx(self, names):
        return [
            mujoco.mj_name2id(
                self.model, type=mujoco.mjtObj.mjOBJ_ACTUATOR, name=name + "_act"
            )
            for name in names
        ]

    @property
    def get_jnt_names(self):
        raise NotImplementedError

    def full_reset(self):
        self.current_steps = 0
