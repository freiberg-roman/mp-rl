"""
Simple tests to test autograd and shapes of network outputs
"""
import torch

from mprl.models.sac.networks import GaussianMPTimePolicy
from torch.optim import Adam


def test_sampling_no_time_no_full_cov_single():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=False, learn_time=False
    ).to(torch.device("cpu"))
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(3,)))
    assert weights.shape == torch.Size([5])
    assert time is None
    assert logp.shape == torch.Size([])


def test_sampling_no_time_no_full_cov_batch():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=False, learn_time=False
    ).to(torch.device("cpu"))
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(100, 3)))
    assert weights.shape == torch.Size([100, 5])
    assert time is None
    assert logp.shape == torch.Size([100])


def test_sampling_time_no_full_cov_single():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=False
    ).to(torch.device("cpu"))
    optimizer = Adam(
        policy_net.parameters(), lr=1.0
    )  # we simply want to notice change in variables
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(3,)))
    assert weights.shape == torch.Size([5])
    assert time.shape == torch.Size([1])
    assert logp.shape == torch.Size([])

    old_val = policy_net.time_scalar.item()
    optimizer.zero_grad()
    (-logp).backward()
    optimizer.step()
    assert old_val != policy_net.time_scalar.item()
    assert policy_net.time_scalar.requires_grad
    assert policy_net.time_scalar.grad.shape == torch.Size([1])


def test_sampling_time_no_full_cov_batch():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=False
    ).to(torch.device("cpu"))
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(100, 3)))
    assert weights.shape == torch.Size([100, 5])
    assert time.shape == torch.Size([100, 1])
    assert logp.shape == torch.Size([100])


def test_sampling_time_full_cov_single():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=True
    ).to(torch.device("cpu"))
    optimizer = Adam(
        policy_net.parameters(), lr=1.0
    )  # we simply want to notice change in variables
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(3,)))
    assert weights.shape == torch.Size([5])
    assert time.shape == torch.Size([1])
    assert logp.shape == torch.Size([])

    old_val = policy_net.log_std_linear.weight.tolist()
    optimizer.zero_grad()
    (-logp).backward()
    optimizer.step()
    assert policy_net.log_std_linear.weight.tolist() != old_val


def test_sampling_time_full_cov_batch():
    policy_net = GaussianMPTimePolicy(
        num_inputs=3, num_weights=5, hidden_dim=10, full_std=True
    ).to(torch.device("cpu"))
    weights, time, logp, _, _ = policy_net.sample(torch.randn(size=(100, 3)))
    assert weights.shape == torch.Size([100, 5])
    assert time.shape == torch.Size([100, 1])
    assert logp.shape == torch.Size([100])
