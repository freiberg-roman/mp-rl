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
                os.path.join(self.work_dir, str(performed_steps), item.store_under)
            )

    def restore_checkpoint(self, performed_steps):
        for item in self.serializable:
            item.load(
                os.path.join(self.work_dir, str(performed_steps), item.store_under)
            )
