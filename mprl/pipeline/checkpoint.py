import os
from typing import List

from mprl.utils.serializable import Serializable


class CheckPoint:
    def __init__(self, serializable: List[Serializable], work_dir):
        self.serializable = serializable
        self.work_dir = work_dir

    def create_checkpoint(self, performed_steps):
        for item in self.serializable:
            item.store(
                item.store_under(
                    os.path.join(self.work_dir, str(performed_steps) + "/")
                )
            )

    def restore_checkpoint(self, dir, performed_steps):
        for item in self.serializable:
            item.load(item.store_under(os.path.join(dir, str(performed_steps) + "/")))
