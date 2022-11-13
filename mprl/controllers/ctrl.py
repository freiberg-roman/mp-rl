from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def get_action(
        self, desired_pos, desired_vel, current_pos, current_vel, action_clip=False
    ):
        pass
