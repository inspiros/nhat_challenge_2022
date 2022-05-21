import torch

import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GraphMaxPool',
    'GraphGlobalMaxPool',
    'PartialGraphMaxPool',
]


def _graph_max_pool(x, p, stride=None):
    # Max pooling of size p. Must be a power of 2.
    if max(p) > 1:
        if stride is None:
            x = F.max_pool2d(x, kernel_size=p)  # B x F x V/p
        else:
            x = F.max_pool2d(x, kernel_size=p, stride=stride)  # B x F x V/p
        return x
    else:
        return x


class GraphMaxPool(nn.Module):
    def __init__(self, graph):
        super(GraphMaxPool, self).__init__()
        self.graph = graph

    def forward(self, x):
        N, C, T, V = x.size()

        out = x.new_empty(N, C, T, len(self.graph.part))
        for i, p in enumerate(self.graph.part):
            num_node = len(p)
            out[..., i:i + 1] = F.max_pool2d(x[..., p], kernel_size=(1, num_node))
        return out


class GraphGlobalMaxPool(nn.Module):
    def forward(self, x):
        out = F.max_pool2d(x, kernel_size=(1, x.size(-1)))
        return out


# noinspection PyMethodOverriding
class PartialGraphMaxPool(GraphMaxPool):
    def __init__(self, graph, multi_channel=True, return_mask=True):
        super(PartialGraphMaxPool, self).__init__(graph)
        self.multi_channel = multi_channel
        self.return_mask = return_mask

    def forward(self, x, mask=None):
        N, C, T, V = x.size()

        out = x.new_empty(N, C, T, len(self.graph.part))
        out_mask = torch.empty_like(out)
        if mask is not None:
            # assign -inf for invalid positions
            x = x.clone()
            x[torch.where(mask == 0)] = -torch.inf
        for i, p in enumerate(self.graph.part):
            values, indices = x[..., p].max(dim=-1, keepdim=True)
            out[..., i:i + 1] = values
            out_mask[..., i:i + 1] = torch.gather(mask[..., p], -1, indices)

        if mask is not None:
            out[torch.where(out_mask == 0)] = 0

        if self.return_mask:
            return out, out_mask
        return out
