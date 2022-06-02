import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Exponential, Independent, MultivariateNormal, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.silu(self.linear1(state))
        x = F.silu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.silu(self.linear1(xu))
        x1 = F.silu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.silu(self.linear4(xu))
        x2 = F.silu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        x = F.silu(self.linear1(state))
        x = F.silu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class GaussianMPTimePolicy(nn.Module):
    def __init__(self, num_inputs, num_weights, hidden_dim, full_std=False):
        super(GaussianPolicy, self).__init__()
        self.full_std = full_std

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_weights)
        self.time_linear = nn.Linear(hidden_dim, 1)
        self.time_scalar = torch.tensor(1.0, requires_grad=True)
        self.sp = nn.Softplus()

        if self.full_std:
            # Learn lower L of log Choleric decomposition
            self.log_std_linear = nn.Linear(
                hidden_dim, (num_weights * (num_weights + 1)) / 2
            )

        self.apply(weights_init_)

    def forward(self, state):
        x = F.silu(self.linear1(state))
        x = F.silu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        time = self.time_scalar * self.sp(self.time_linear(x))
        return mean, log_std, time

    def sample(self, state):
        mean, log_std, t = self.forward(state)
        std = log_std.exp()
        if self.full_std:
            normal = MultivariateNormal(mean, scale_tril=std)
        else:
            normal = Independent(Normal(mean, std))
        weights = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        exp_dist = Exponential(t)  # time is sampled from an exponential distribution
        time = exp_dist.rsample()
        log_prob = exp_dist.log_prob(time) + normal.log_prob(weights)
        return weights, time, log_prob, mean, t

    def to(self, device):
        return super(GaussianPolicy, self).to(device)
