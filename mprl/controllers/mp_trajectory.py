import numpy as np
import torch as ch
from mp_pytorch.mp import ProDMP
from mp_pytorch.util import tensor_linspace

from mprl.utils.ds_helper import to_ts
from mprl.utils.math_helper import build_lower_matrix


class MPTrajectory:
    def __init__(self, dt: float, mp: ProDMP, num_steps: int):
        self.mp: ProDMP = mp
        self.dt = dt
        self.current_traj: ch.Tensor = None
        self.current_traj_v: ch.Tensor = None
        self.current_t: int = 0
        self.num_t: int = 0
        self.num_steps: int = num_steps

    def init(
        self,
        weights: ch.Tensor,
        bc_pos: np.ndarray,
        bc_vel: np.ndarray,
    ) -> "MPTrajectory":
        t = self.dt * self.num_steps
        times = tensor_linspace(0, t, self.num_steps + 1).unsqueeze(dim=0)
        bc_pos = to_ts(bc_pos)
        bc_vel = to_ts(bc_vel)
        bc_time = ch.tensor([0.0] * weights.shape[0])
        self.current_traj = self.mp.get_traj_pos(
            times=times,
            params=weights,
            bc_time=bc_time,
            bc_pos=bc_pos,
            bc_vel=bc_vel,
        ).squeeze()
        self.current_traj_v = self.mp.get_traj_vel().squeeze()
        self.current_t = 0
        return self

    def __next__(self):
        if self.current_traj is None or self.current_t >= self.num_steps:
            raise StopIteration

        q, v = (
            self.current_traj[..., self.current_t, :],
            self.current_traj_v[..., self.current_t, :],
        )
        self.current_t += 1
        return q, v

    def __iter__(self):
        return self

    def __getitem__(self, item):
        if self.current_traj is None or self.current_traj_v is None:
            return None, None
        q, v = (
            self.current_traj[..., item, :],
            self.current_traj_v[..., item, :],
        )
        return q, v

    def get_current(self):
        return self.current_traj[..., self.current_t - 1, :]

    def get_next_bc(self):
        if self.current_traj is None or self.current_traj_v is None:
            return None, None
        q, v = (
            self.current_traj[..., -1, :],
            self.current_traj_v[..., -1, :],
        )
        return q, v

    def reset_planner(self):
        self.current_traj = None
        self.current_traj_v = None
        self.current_t = 0
        self.num_t = 0

    @property
    def steps_planned(self):
        return len(self.current_traj) - 1

    def get_times(self, num_t):
        return tensor_linspace(0, self.dt * num_t, num_t + 1).unsqueeze(dim=0)

    def get_log_prob(self, smp_traj, mean, chol_cov, bc_q, bc_v):
        num_t = smp_traj.shape[1] - 1
        bc_time = ch.tensor([0.0] * mean.shape[0], device=self.device)
        times = tensor_linspace(0, self.dt * num_t, num_t + 1).unsqueeze(dim=0)
        chol_cov = build_lower_matrix(chol_cov)
        self.mp.update_inputs(
            times=times,
            params=mean,
            params_L=chol_cov,
            bc_time=bc_time,
            bc_pos=bc_q,
            bc_vel=bc_v,
        )
        traj_mean = self.mp.get_traj_pos(flat_shape=True)
        traj_cov = self.mp.get_traj_pos_cov()
        traj = smp_traj.flatten(start_dim=-2, end_dim=-1)
        mv = ch.distributions.MultivariateNormal(
            loc=traj_mean, covariance_matrix=traj_cov, validate_args=False
        )
        return mv.log_prob(traj)
