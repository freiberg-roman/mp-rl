"""
    Utilities of data type and structure
"""
from typing import List, Literal, Tuple, Union

import numpy as np
import torch


def make_iterable(
    data: any, default: Literal["tuple", "list"] = "tuple"
) -> Union[Tuple, List]:
    """
    Make data a tuple or list, i.e. (data) or [data]
    Args:
        data: some data
        default: default type
    Returns:
        (data) if it is not a tuple
    """
    if isinstance(data, (tuple, list)):
        return data
    else:
        if default == "tuple":
            return (data,)  # Do not use tuple()
        elif default == "list":
            return [
                data,
            ]
        else:
            raise NotImplementedError


def to_np(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Transfer any type and device of tensor to a numpy ndarray
    Args:
        tensor: np.ndarray, cpu tensor or gpu tensor

    Returns:
        tensor in np.ndarray
    """
    if is_np(tensor):
        return tensor
    elif is_ts(tensor):
        if tensor.device.type == "cpu":
            return tensor.detach().numpy()
        elif tensor.device.type == "cuda":
            return tensor.detach().cpu().numpy()
    raise NotImplementedError


def to_nps(*tensors: [Union[np.ndarray, torch.Tensor]]) -> [np.ndarray]:
    """
    transfer a list of any type of tensors to np.ndarray
    Args:
        tensors: a list of tensors

    Returns:
        a list of np.ndarray
    """
    return [to_np(tensor) for tensor in tensors]


def is_np(data: any) -> bool:
    """
    is data a numpy array?
    """
    return isinstance(data, np.ndarray)


def to_ts(
    data: Union[int, float, np.ndarray, torch.Tensor],
    dtype: Literal["float32", "float64"] = "float32",
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.Tensor:
    """
    Transfer any numerical input to a torch tensor in default data type + device

    Args:
        device: device of the tensor, default: cpu
        dtype: data type of tensor, float 32 or float 64 (double)
        data: float, np.ndarray, torch.Tensor

    Returns:
        tensor in torch.Tensor
    """
    if dtype == "float32":
        data_type = torch.float32
    elif dtype == "float64":
        data_type = torch.float64
    else:
        raise NotImplementedError

    if isinstance(data, (float, int)):
        return torch.tensor(data, dtype=data_type, device=device)
    elif is_ts(data):
        return data

    elif is_np(data):
        return torch.tensor(data, dtype=data_type, device=device)
    else:
        raise NotImplementedError


def to_tss(
    *datas: [Union[int, float, np.ndarray, torch.Tensor]],
    dtype: Literal["float32", "float64"] = "float32",
    device: Literal["cpu", "cuda"] = "cpu"
) -> [torch.Tensor]:
    """
    transfer a list of any type of numerical input to a list of tensors in given
    data type and device

    Args:
        datas: a list of data
        dtype: data type of tensor, float 32 or float 64 (double)
        device: device of the tensor, default: cpu

    Returns:
        a list of np.ndarray
    """
    return [to_ts(data, dtype, device) for data in datas]


def is_ts(data: any) -> bool:
    """
    is data a torch Tensor?
    """
    return isinstance(data, torch.Tensor)


def compare_networks(model1, model2):
    return all(
        not p1.data.ne(p2.data).sum() > 0
        for p1, p2 in zip(model1.parameters(), model2.parameters())
    )
