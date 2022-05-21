from typing import Union, Any

import numpy as np
import torch

__all__ = [
    'to_tensor',
    'tensor_slice',
    'broadcast_dims',
]


def to_tensor(x: Any, dtype=None, device=None):
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def tensor_slice(ndim: int,
                 dim: int,
                 indices: Union[slice, int] = slice(None)):
    slices = [slice(None)] * ndim
    slices[dim] = indices
    return slices


def broadcast_dims(ndim: int,
                   dim: int):
    if dim < 0:
        dim = ndim + dim
    shape = [1] * ndim
    shape[dim] = -1
    return shape
