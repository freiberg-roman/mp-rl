import numpy as np
import torch
from mp_pytorch.mp import ProDMP
from mp_pytorch.util import tensor_linspace

from mprl.utils.ds_helper import to_ts


class MPTrajectory:
    def __init__(self, dt: float, mp: ProDMP, device: torch.device):
        self.mp: ProDMP = mp
        self.dt = dt
        self.current_traj: torch.Tensor = None
        self.current_traj_v: torch.Tensor = None
        self.current_t: int = 0
        self.device: torch.device = device
        self.num_t: int = 0

    def init(
        self,
        weights: torch.Tensor,
        bc_pos: np.ndarray,
        bc_vel: np.ndarray,
        num_t: int,
    ) -> "MPTrajectory":
        t = self.dt * num_t
        times = tensor_linspace(0, t, num_t + 1).unsqueeze(dim=0)
        bc_pos = to_ts(bc_pos, device=self.device)
        bc_vel = to_ts(bc_vel, device=self.device)
        bc_time = torch.tensor([0.0] * weights.shape[0], device=self.device)
        self.current_traj = self.mp.get_traj_pos(
            times=times,
            params=weights,
            bc_time=bc_time,
            bc_pos=bc_pos,
            bc_vel=bc_vel,
        ).squeeze()
        self.current_traj_v = (
            self.current_traj[..., 1:, :] - self.current_traj[..., :-1, :]
        ) / self.dt
        self.current_t = 0
        self.num_t = num_t
        return self

    def __next__(self):
        if self.current_traj is None or self.current_t >= self.num_t:
            raise StopIteration

        q, v = (
            self.current_traj[..., self.current_t + 1, :],
            self.current_traj_v[..., self.current_t, :],
        )
        self.current_t += 1
        return q, v

    def __iter__(self):
        return self

    def __getitem__(self, item):
        q, v = (
            self.current_traj[..., item + 1, :],
            self.current_traj_v[..., item, :],
        )
        return q, v

    def get_current(self):
        return self.current_traj[..., self.current_t - 1, :]

    def reset_planner(self):
        self.current_traj = None
        self.current_traj_v = None
        self.current_t = 0
        self.num_t = 0

    @property
    def steps_planned(self):
        return len(self.current_traj) - 1
