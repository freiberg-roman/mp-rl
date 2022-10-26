#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch as ch


def torch_batched_trace(x) -> ch.Tensor:
    """
    Compute trace in n,m of batched matrix
    Args:
        x: matrix with shape [a,...l, n, m]

    Returns: trace with shape [a,...l]

    """
    return ch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def tensorize(x, cpu=True, dtype=ch.float32):
    """
    Utility function for turning arrays into tensors
    Args:
        x: data
        cpu: Whether to generate a CPU or GPU tensor
        dtype: dtype of tensor

    Returns:
        gpu/cpu tensor of x with specified dtype
    """
    return cpu_tensorize(x, dtype) if cpu else gpu_tensorize(x, dtype)


def gpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cuda tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        gpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype).cuda()


def cpu_tensorize(x, dtype=None):
    """
    Utility function for turning arrays into cpu tensors
    Args:
        x: data
        dtype: dtype to generate

    Returns:
        cpu tensor of x
    """
    dtype = dtype if dtype else x.dtype
    return ch.tensor(x).type(dtype)


def get_numpy(x):
    """
    Convert torch tensor to numpy
    Args:
        x: torch.Tensor

    Returns:
        numpy tensor of x

    """
    return x.cpu().detach().numpy()


def inverse_softplus(x):
    """
    x = inverse_softplus(softplus(x))
    Args:
        x: data

    Returns:

    """
    return (x.exp() - 1.0).log()
