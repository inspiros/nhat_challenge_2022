import numpy as np
import torch
import torch.nn as nn

__all__ = [
    'WithMaskCompose',
    'RandomMaskKeypoint',
    'RandomMaskKeypointBetween'
]


class WithMaskCompose(nn.Module):

    def __init__(self, transforms):
        super(WithMaskCompose, self).__init__()
        self.transforms = transforms

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class WithMask(torch.nn.Module):
    def forward(self, x, mask=None):
        out_mask = torch.ones_like(x)
        if mask is not None:
            out_mask = out_mask * mask
        return x, out_mask


class RandomMaskKeypoint(torch.nn.Module):

    def __init__(self, p, temporal_p=1.):
        super(RandomMaskKeypoint, self).__init__()
        self.p = p
        self.temporal_p = temporal_p

    def forward(self, x, mask):
        C, T, V = x.size()

        temporal_mask = torch.rand(T).lt(self.temporal_p)
        spatial_mask = torch.rand(T, V).lt(self.p)
        out_mask = torch.einsum('t,tv->tv', temporal_mask, spatial_mask)
        out_mask = out_mask.unsqueeze(0).repeat_interleave(2, 0)
        out_mask = out_mask.logical_not().type_as(x)

        out_mask = out_mask * mask
        return x, out_mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p:.03f}, temporal_p={self.temporal_p})"


class RandomMaskKeypointBetween(torch.nn.Module):

    def __init__(self, low=None, high=None, temporal_p=1.):
        super(RandomMaskKeypointBetween, self).__init__()
        self.low = low
        self.high = high
        self.temporal_p = temporal_p

    def forward(self, x, mask):
        C, T, V = x.size()
        low = 0 if self.low is None else max(0, self.low)
        high = V if self.high is None else min(self.high, V)

        out_mask = torch.ones_like(x)
        temporal_mask = torch.rand(T).gt(self.temporal_p)
        for t in torch.where(~temporal_mask)[0]:
            masked_inds = np.random.choice(V, np.random.randint(low, high + 1), replace=False)
            out_mask[:, t, masked_inds] = 0

        out_mask = out_mask * mask
        return x, out_mask

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high}, temporal_p={self.temporal_p})"
