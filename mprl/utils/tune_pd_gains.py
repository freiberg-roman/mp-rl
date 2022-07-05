import pathlib

import numpy as np
from omegaconf import OmegaConf

from mprl.controllers.pd_controller import PDController
from mprl.env.mujoco import ReacherEnv
from mprl.utils.ds_helper import to_np

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"

if __name__ == "__main__":
    pgains = [1.0] * 2
    dgains = [0.1] * 2
    cfg = OmegaConf.create({"pgains": pgains, "dgains": dgains})
    ctrl = PDController(cfg)
    env = ReacherEnv(base=BASE)

    qpos_start = [1.085, 0.305]
    qpos_end = [-1.5, -1.5]
    qpos_desired = np.linspace(qpos_start, qpos_end, num=200)
    qvel_desired = (qpos_desired[1:] - qpos_desired[:-1]) / env.dt

    env.reset()
    env.set_robot_to(qpos_start)
    for i, qv in enumerate(zip(qpos_desired[1:], qvel_desired)):
        q, v = qv
        raw_action, _ = ctrl.get_action(
            q, v, env.data.qpos.flat[:2], env.data.qvel.flat[:2]
        )
        act = to_np(raw_action)
        state, _, _, _ = env.step(act)
        q_c, v_c = env.decompose(state)
        env.render()

        if i % 50 == 0:
            print("delta in position: ", np.mean((q - q_c) ** 2))
            print("delta in velocity: ", np.mean((v - v_c) ** 2))
