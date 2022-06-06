import numpy as np
import torch
from mp_pytorch import MPFactory, ProMP, tensor_linspace
from omegaconf import OmegaConf, open_dict


class MPTrajectory:
    def __init__(self, cfg: OmegaConf):
        with open_dict(cfg):
            cfg.mp_type = "idmp"
        self.mp = MPFactory.init_mp(cfg)
        self.dt = cfg["mp_args"]["dt"]
        self.current_traj = None
        self.current_traj_v = None
        self.current_t = 0
        self.device = torch.device(cfg.device)

    def re_init(self, weight_time, bc_pos, bc_vel, t=None):
        if t is None:
            t = weight_time[-1].item()
        num_t = int(t / self.dt) + 1
        if num_t == 1:
            num_t += 1

        times = tensor_linspace(0, t, num_t).unsqueeze(dim=0)
        bc_pos = torch.from_numpy(np.expand_dims(bc_pos, axis=0)).to(self.device)
        bc_vel = torch.from_numpy(np.expand_dims(bc_vel, axis=0)).to(self.device)
        weights = weight_time[:-1].unsqueeze(dim=0)
        bc_time = torch.tensor([0.0], device=self.device)
        self.current_traj = self.mp.get_traj_pos(
            times=times,
            params=weights,
            bc_time=bc_time,
            bc_pos=bc_pos,
            bc_vel=bc_vel,
        ).squeeze()
        self.current_traj_v = (
            self.current_traj[1:, ...] - self.current_traj[:-1, ...]
        ) / self.dt
        self.current_t = 0
        return self

    def __next__(self):
        if self.current_traj is None or self.current_t >= len(self.current_traj) - 1:
            raise StopIteration

        q, v = self.current_traj[self.current_t], self.current_traj_v[self.current_t]
        self.current_t += 1
        return q, v

    def __iter__(self):
        return self

    @property
    def steps_planned(self):
        return len(self.current_traj) - 1
