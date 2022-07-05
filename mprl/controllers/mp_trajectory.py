from typing import Optional, Tuple

import numpy as np
import torch
from mp_pytorch import IDMP, MPFactory, tensor_linspace
from omegaconf import DictConfig, open_dict

from mprl.utils.ds_helper import to_ts


class MPTrajectory:
    def __init__(self, cfg: DictConfig):
        with open_dict(cfg):
            cfg.mp_type = "idmp"
        self.mp: IDMP = MPFactory.init_mp(cfg)
        self.dt = cfg["mp_args"]["dt"]
        self.current_traj: torch.Tensor = None
        self.current_traj_v: torch.Tensor = None
        self.current_t: int = 0
        self.device: torch.device = torch.device(cfg.device)

    def re_init(
        self,
        weight_time: torch.Tensor,
        bc_pos: np.ndarray,
        bc_vel: np.ndarray,
        num_t: Optional[int] = None,
    ) -> "MPTrajectory":
        if num_t is None:
            t = weight_time[:, -1].item()
            weights = weight_time[:, :-1]
            num_t = int((t + self.dt) / self.dt)
        else:
            t = self.dt * num_t
            weights = weight_time

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
        return self

    def __next__(self):
        if self.current_traj is None or self.current_t >= len(self.current_traj) - 1:
            raise StopIteration

        q, v = (
            self.current_traj[..., self.current_t + 1, :],
            self.current_traj_v[..., self.current_t, :],
        )
        self.current_t += 1
        return q, v

    def __iter__(self):
        return self

    def reset_planner(self):
        self.current_traj = None
        self.current_traj_v = None
        self.current_t = 0

    def one_step_ctrl(
        self, weights: torch.Tensor, bc_pos: torch.Tensor, bc_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        times = tensor_linspace(0, self.dt, 2).unsqueeze(dim=0)
        bc_time = torch.tensor([0.0] * weights.shape[0], device=self.device)
        current_traj = self.mp.get_traj_pos(
            times=times,
            params=weights,
            bc_time=bc_time,
            bc_pos=bc_pos,
            bc_vel=bc_vel,
        )
        current_traj_v = (
            current_traj[:, 1:, ...] - current_traj[:, :-1, ...]
        ) / self.dt
        return (
            current_traj[:, 1, :].squeeze(),
            current_traj_v[:, 0, :].squeeze(),
        )

    @property
    def steps_planned(self):
        return len(self.current_traj) - 1
