from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.optim import Adam

from mprl.models.sac_common.critic_loss import sac_critic_loss
from mprl.models.sac_common.policy_loss import sac_policy_loss
from mprl.utils import EnvSteps
from mprl.utils.math_helper import soft_update


class SACUpdate:
    def __init__(
        self,
        critic_loss: Optional[Callable[[any, EnvSteps], torch.tensor]] = None,
        policy_loss: Optional[Callable[[any, EnvSteps], torch.tensor]] = None,
    ):
        self.times_called = 0

        # Critic loss
        if critic_loss is None:
            self.critic_loss = sac_critic_loss
        else:
            self.critic_loss = critic_loss

        # Policy loss
        if policy_loss is None:
            self.policy_loss = sac_policy_loss
        else:
            self.policy_loss = policy_loss

    def __call__(
        self,
        agent: any,
        optimizer_policy: Adam,
        optimizer_critic: Adam,
        batch: EnvSteps,
    ) -> dict:
        # Update critic
        qf_loss = self.critic_loss(agent, batch)
        optimizer_critic.zero_grad()
        qf_loss.backward()
        optimizer_critic.step()

        # Update policy
        policy_loss = self.policy_loss(agent, batch)
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update target networks
        soft_update(agent.critic_target, agent.critic, agent.tau)

        self.times_called += 1
        return {
            "sac_update_times_called": self.times_called,
            "qf_loss": qf_loss.item(),
            "policy_loss": policy_loss.item(),
        }
