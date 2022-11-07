from pathlib import Path

import torch as ch
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
        state_dim_in: int,
        action_dim: int,
        state_dim_out: int,
        num_experts: int,
        network_width: int,
        variance: float = 1.0,
        lr: float = 3e-4,
        use_batch_normalization: bool = False,
    ):
        super(MixtureOfExperts, self).__init__()

        if use_batch_normalization:
            self.bn_in = nn.BatchNorm1d(state_dim_in)
        self.linear1 = nn.Linear(
            state_dim_in + action_dim,
            network_width,
        )
        self.linear2 = nn.Linear(network_width, network_width)
        self.expert_heads = []

        for _ in range(num_experts):
            self.expert_heads.append(nn.Linear(network_width, state_dim_out))

        self.expert_heads = nn.ModuleList(self.expert_heads)
        self._action_dim = action_dim
        self.optimizer: ch.optim.Optimizer = ch.optim.Adam(
            self.parameters(),
            lr=lr,
        )
        self.variance = variance
        self.use_batch_normalization = use_batch_normalization

    def forward(self, state, action):
        if self.use_batch_normalization:
            state = self.bn_in(state)
        net_in = ch.cat([state, action], dim=-1)
        x1 = F.silu(self.linear1(net_in))
        x2 = F.silu(self.linear2(x1))
        means = []
        for head in self.expert_heads:
            means.append(head(x2))
        means = ch.stack(means, 1)

        ind_expert_dist = Independent(
            MultivariateNormal(means, self.variance * ch.eye(means.size(dim=-1))), 0
        )
        categorical_experts = Categorical(probs=ch.ones(len(self.expert_heads)))
        return MixtureSameFamily(categorical_experts, ind_expert_dist)

    def log_prob(self, state, action, next_state_delta):
        prob = self.forward(state, action)
        return prob.log_prob(next_state_delta)

    def loss(self, state, action, next_state_delta):
        log_prob = self.log_prob(state, action, next_state_delta)
        return (-log_prob).mean()  # NLL

    @ch.no_grad()
    def next_state_delta(self, states, actions, deterministic=False):
        states = to_ts(states)
        actions = to_ts(actions)
        if not deterministic:
            pred_delta = self.forward(states, actions).sample()
        else:
            pred_delta = self.forward(states, actions).mean
        return pred_delta

    def update(self, states: ch.Tensor, actions: ch.Tensor, next_states: ch.Tensor):
        loss = self.loss(states, actions, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"model_loss": loss.item()}

    def save(self, base_path, folder):
        path = base_path + folder + "/moe/"
        Path(path).mkdir(parents=True, exist_ok=True)
        ch.save(self.state_dict(), path + "model.pt")

    def load(self, path, train=True):
        path = path + "moe/model.pt"
        self.load_state_dict(ch.load(path))
        if train:
            self.train()
        else:
            self.eval()
