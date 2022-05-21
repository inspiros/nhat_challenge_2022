import numpy as np
import torch

from utils.tensor_utils import tensor_slice

__all__ = [
    'Permute',
    'FrameNormalize',
    'FrameDenormalize',
    'RandomJitter',
    'RandomDrop',
    'RandomDropDim',
]


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, tensor):
        return torch.permute(tensor, self.dims)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dims})"


class FrameNormalize(torch.nn.Module):
    """
    Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    """

    def __init__(self, frame_width, frame_height, keep_aspect_ratio=False, dim=-1, inline=True):
        super(FrameNormalize, self).__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.dim = dim
        self.inline = inline

    def forward(self, tensor):
        if not self.inline:
            tensor = tensor.clone()
        if self.keep_aspect_ratio:
            tensor = tensor / self.frame_width * 2
            tensor[tensor_slice(tensor.ndim, self.dim, 0)] -= 1
            tensor[tensor_slice(tensor.ndim, self.dim, 1)] -= self.frame_height / self.frame_width
        else:
            tensor[tensor_slice(tensor.ndim, self.dim, 0)] = \
                tensor[tensor_slice(tensor.ndim, self.dim, 0)] / self.frame_width * 2 - 1
            tensor[tensor_slice(tensor.ndim, self.dim, 1)] = \
                tensor[tensor_slice(tensor.ndim, self.dim, 1)] / self.frame_height * 2 - 1
        return tensor

    def inverse(self):
        return self.__invert__()

    def __invert__(self):
        return FrameDenormalize(self.frame_width, self.frame_height, self.keep_aspect_ratio, self.dim, self.inline)

    def __repr__(self):
        res = f"{self.__class__.__name__}("
        res += f"frame_size={(self.frame_width, self.frame_height)}"
        if self.keep_aspect_ratio:
            res += f"keep_aspect_ratio={self.keep_aspect_ratio}"
        res += ")"
        return res


class FrameDenormalize(torch.nn.Module):
    """
    Inverse of FrameNormalize
    """

    def __init__(self, frame_width, frame_height, keep_aspect_ratio=False, dim=-1, inline=True):
        super(FrameDenormalize, self).__init__()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.keep_aspect_ratio = keep_aspect_ratio
        self.dim = dim
        self.inline = inline

    def forward(self, tensor):
        if not self.inline:
            tensor = tensor.clone()
        if self.keep_aspect_ratio:
            tensor[tensor_slice(tensor.ndim, self.dim, 0)] += 1
            tensor[tensor_slice(tensor.ndim, self.dim, 1)] += self.frame_height / self.frame_width
            tensor = tensor / 2 * self.frame_width
        else:
            tensor[tensor_slice(tensor.ndim, self.dim, 0)] = \
                (tensor[tensor_slice(tensor.ndim, self.dim, 0)] + 1) / 2 * self.frame_width
            tensor[tensor_slice(tensor.ndim, self.dim, 1)] = \
                (tensor[tensor_slice(tensor.ndim, self.dim, 1)] + 1) / 2 * self.frame_height
        return tensor

    def inverse(self):
        return self.__invert__()

    def __invert__(self):
        return FrameNormalize(self.frame_width, self.frame_height, self.keep_aspect_ratio, self.dim, self.inline)

    def __repr__(self):
        res = f"{self.__class__.__name__}("
        res += f"frame_size={(self.frame_width, self.frame_height)}"
        if self.keep_aspect_ratio:
            res += f"keep_aspect_ratio={self.keep_aspect_ratio}"
        res += ")"
        return res


class RandomJitter(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super(RandomJitter, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        return tensor + torch.normal(self.mean, self.std, tensor.size())

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean:.03f}, std={self.std:.03f})"


class RandomDrop(torch.nn.Module):
    def __init__(self, p, value=0):
        super(RandomDrop, self).__init__()
        self.p = p
        self.value = value

    def forward(self, tensor):
        tensor_view = tensor.flatten(0)
        tensor_view[torch.rand(tensor.numel()) < self.p] = self.value
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p:.03f})"


class RandomDropDim(torch.nn.Module):

    def __init__(self, p, value=0, dim=-1):
        super(RandomDropDim, self).__init__()
        self.p = p
        self.value = value
        self.dim = dim

    def forward(self, tensor):
        dim = self.dim if self.dim >= 0 else tensor.ndim + self.dim
        if not 0 <= dim < tensor.ndim:
            raise ValueError(f'dim={self.dim} is out of range.')

        other_dims = list(tensor.size())
        other_dims.pop(dim)
        mask = torch.rand(np.prod(other_dims)).lt(self.p).view(other_dims)
        mask.unsqueeze_(dim)
        mask = torch.repeat_interleave(mask, tensor.size(dim), dim)

        tensor[mask] = self.value
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p:.03f})"
