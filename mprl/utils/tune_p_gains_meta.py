import pathlib

import numpy as np
from matplotlib import pyplot as plt

from mprl.controllers.meta_controller import MetaController
from mprl.env.mujoco.meta.adapters import OriginalMetaWorld
from mprl.utils.ds_helper import to_np

BASE = str(pathlib.Path(__file__).parent.resolve()) + "/../../resources/"

if __name__ == "__main__":
    ctrl = MetaController(pgains=np.array([35] * 3 + [-10]))
    env = OriginalMetaWorld("reach-v2")

    state, _ = env.reset(time_out_after=500)
    xyz_start = state[:4]
    xyz_end = np.array([0.4, 0.8, 0.12, 0.5])

    xyz_des = np.linspace(xyz_start, xyz_end, num=501)

    xyz_curr = []
    for i, xyz in enumerate(xyz_des[1:, ...]):
        trq = ctrl.get_action(xyz, None, state[:4], None)
        act = to_np(trq)
        state, _, _, _, _, _ = env.step(act)
        xyz_curr.append(state[:4])
        env.render()

        if i % 50 == 0:
            print("delta in position: ", np.mean((xyz - state[:4]) ** 2))

    for i in range(4):
        t = np.linspace(0, 6.25, 500)
        a = xyz_des[1:, i]
        b = np.array(xyz_curr)[:, i]

        plt.plot(t, a, "r")  # plotting t, a separately
        plt.plot(t, b, "b")  # plotting t, b separately
        plt.show()
