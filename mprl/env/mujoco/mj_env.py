import mujoco
import numpy as np

from .rendering.mj_rendering import RenderContextOffscreen, Viewer

DEFAULT_SIZE = 480


class MujocoEnv:
    def __init__(self, path, frame_skip):

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}
        self.time_out_after = None
        self.current_steps = 0
        self._total_steps = 0

        self.frame_skip = frame_skip

        self.viewer = None

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.low, self.high = self._set_action_space()

    def _set_action_space(self):
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

    def reset(self, time_out_after=None):
        self.time_out_after = time_out_after
        self.current_steps = 0
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
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
        self.data.ctrl[:] = ctrl
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
