from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

from mprl.models import Predictable
from mprl.utils.ds_helper import to_ts


class MixtureOfExperts(nn.Module, Predictable):
    def __init__(
        self,
        cfg,
        prep_input_fn=lambda x: x,
    ):
        super(MixtureOfExperts, self).__init__()

        self.bn_in = nn.BatchNorm1d(cfg.env.state_dim)
        self.bb_out = nn.BatchNorm1d(cfg.env.state_dim)
        self._prep_input_fn = prep_input_fn
        self.linear1 = nn.Linear(
            cfg.env.state_dim + cfg.env.action_dim,
            cfg.hidden_dim,
        )
        self.linear2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.expert_log_prob_contribution = nn.Linear(cfg.hidden_dim, cfg.num_experts)
        self.expert_heads = []

        for _ in range(cfg.num_experts):
            self.expert_heads.append(nn.Linear(cfg.hidden_dim, cfg.env.state_dim))

        self.expert_heads = nn.ModuleList(self.expert_heads)
        self._action_dim = cfg.env.action_dim
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters())

    def forward(self, state, action):
        n_state = self.bn_in(self._prep_input_fn(state))
        net_in = torch.cat([n_state, action], 1)

        x1 = F.silu(self.linear1(net_in))
        x2 = F.silu(self.linear2(x1))

        expert_log_prob = self.expert_log_prob_contribution(x2)
        categorical_experts = Categorical(logits=expert_log_prob)

        means = []
        for head in self.expert_heads:
            means.append(head(x2))
        means = torch.stack(means, 1)

        ind_expert_dist = Independent(
            MultivariateNormal(means, torch.eye(means.size(dim=-1))), 0
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
    def next_state(self, states, actions, sample_n=1, deterministic=False):
        states = to_ts(states)
        actions = to_ts(actions)
        if not deterministic:
            pred_delta = self.forward(states, actions).sample((sample_n,))
        else:
            pred_delta = self.forward(states, actions).mean

        # denorm pred_delta
        pred_delta = (
            pred_delta * torch.sqrt(self.bb_out.running_var + self.bb_out.eps)
            + self.bb_out.running_mean
        )
        return (states + pred_delta)[0, ...]

    def update_parameters(self, batch):
        (
            states,
            next_states,
            actions,
            _,
            _,
        ) = batch.to_torch_batch()
        loss = self.loss(states, actions, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
