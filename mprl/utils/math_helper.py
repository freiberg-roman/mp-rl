from typing import Optional

import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def truncated_normal(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor


def build_lower_matrix(
    param_diag: torch.Tensor, param_off_diag: Optional[torch.Tensor]
) -> torch.Tensor:
    """
    Compose the lower triangular matrix L from diag and off-diag elements
    It seems like faster than using the cholesky transformation from PyTorch
    Args:
        param_diag: diagonal parameters
        param_off_diag: off-diagonal parameters

    Returns:
        Lower triangular matrix L

    """
    dim_pred = param_diag.shape[-1]
    # Fill diagonal terms
    L = param_diag.diag_embed()
    if param_off_diag is not None:
        # Fill off-diagonal terms
        [row, col] = torch.tril_indices(dim_pred, dim_pred, -1)
        L[..., row, col] = param_off_diag[..., :]

    return L
