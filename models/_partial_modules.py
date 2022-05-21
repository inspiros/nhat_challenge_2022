import torch
import torch.nn as nn

__all__ = [
    'Partial',
    'PartialSequential',
    'PartialDropout',
    'PartialSum',
    'PartialCat',
]


class Partial(nn.Module):
    def __init__(self, module):
        super(Partial, self).__init__()
        self.module = module

    def forward(self, x, mask):
        x = self.module(x)
        return x, mask


# noinspection PyMethodOverriding
class PartialSequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self:
            x, mask = module(x, mask)
        return x, mask


# noinspection PyMethodOverriding
class PartialDropout(nn.Dropout):
    def forward(self, x, mask):
        if not self.training:
            return x, mask
        if not self.inplace:
            x = x.clone()
            mask = mask.clone()
        drop_mask = torch.rand(x.numel()).lt(self.p).view_as(x)
        x[drop_mask] = 0
        mask[drop_mask] = 0
        return x, mask


class PartialSum(nn.Module):
    def forward(self, xs, masks):
        return sum(xs), sum(masks).bool().float()


class PartialCat(nn.Module):
    def forward(self, xs, masks):
        return torch.cat(xs), torch.cat(masks)
