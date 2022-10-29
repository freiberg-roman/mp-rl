from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

from mprl.utils.ds_helper import to_ts


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        state_dim_in,
        action_dim,
        state_dim_out,
        num_experts,
        network_width,
        variance=1.0,
    ):
        super(MixtureOfExperts, self).__init__()

        self.bn_in = nn.BatchNorm1d(state_dim_in)
        self.bb_out = nn.BatchNorm1d(state_dim_out)
        self.linear1 = nn.Linear(
            state_dim_in + action_dim,
            network_width,
        )
        self.linear2 = nn.Linear(network_width, network_width)
        self.expert_log_prob_contribution = nn.Linear(network_width, num_experts)
        self.expert_heads = []

        for _ in range(num_experts):
            self.expert_heads.append(nn.Linear(network_width, state_dim_out))

        self.expert_heads = nn.ModuleList(self.expert_heads)
        self._action_dim = action_dim
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters())
        self.variance = variance

    def forward(self, state, action):
        n_state = self.bn_in(self._prep_input_fn(state))
        net_in = torch.cat([n_state, action], dim=-1)

        x1 = F.silu(self.linear1(net_in))
        x2 = F.silu(self.linear2(x1))

        expert_log_prob = self.expert_log_prob_contribution(x2)
        categorical_experts = Categorical(logits=expert_log_prob)

        means = []
        for head in self.expert_heads:
            means.append(head(x2))
        means = torch.stack(means, 1)

        ind_expert_dist = Independent(
            MultivariateNormal(means, self.variance * torch.eye(means.size(dim=-1))), 0
        )
        return MixtureSameFamily(categorical_experts, ind_expert_dist)

    def log_prob(self, state, action, next_state):
        n_next_state_delta = self.bb_out(next_state - state)
        prob = self.forward(state, action)
        return prob.log_prob(n_next_state_delta)

    def loss(self, state, action, next_state):
        log_prob = self.log_prob(state, action, next_state)
        return (-log_prob).mean()  # NLL

    @torch.no_grad()
    def next_state(self, states, actions, deterministic=False):
        states = to_ts(states)
        actions = to_ts(actions)
        if not deterministic:
            pred_delta = self.forward(states, actions).sample((states.shape[:-1], 1))
        else:
            pred_delta = self.forward(states, actions).mean

        # denorm pred_delta
        pred_delta = (
            pred_delta * torch.sqrt(self.bb_out.running_var + self.bb_out.eps)
            + self.bb_out.running_mean
        )
        return states + pred_delta

    def update(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ):
        loss = self.loss(states, actions, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"model_loss": loss.item()}

    def save(self, base_path, folder):
        path = base_path + folder + "/moe/"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path + "model.pt")

    def load(self, path, train=True):
        path = path + "moe/model.pt"
        self.load_state_dict(torch.load(path))
        if train:
            self.train()
        else:
            self.eval()
