import torch
from mp_pytorch import MPFactory, ProMP, tensor_linspace
from omegaconf import OmegaConf


class MPTrajectory:
    def __init__(self, cfg: OmegaConf):
        cfg.mp_type = "promp"
        self.mp = MPFactory.init_mp(cfg)
        self.dt = cfg["mp_args"]["dt"]
        self.current_traj = None
        self.current_traj_v = None
        self.current_t = 0

    def re_init(self, mean, L, t, bc_pos, bc_vel):
        num_t = int(t / self.dt) + 1
        if num_t == 1:
            num_t += 1

        times = tensor_linspace(0, t, num_t).unsqueeze(-1)
        bc_time = times[:, 0]
        self.mp.update_mp_inputs(
            times=times,
            params=mean,
            params_L=L,
            bc_time=bc_time,
            bc_pos=bc_pos,
            bc_vel=bc_vel,
        )
        self.current_traj = self.mp.sample_trajectories(num_smp=1)[0]
        self.current_traj_v = (self.current_traj[1:] - self.current_t[:-1]) / self.dt
        return self

    def __next__(self):
        if self.current_traj is None or self.current_t >= len(self.current_traj) - 1:
            raise StopIteration

        q, v = self.current_traj[self.current_t], self.current_traj_v[self.current_t]
        self.current_t += 1
        return q, v

    def __iter__(self):
        return self
