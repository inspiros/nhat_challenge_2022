import torch.nn as nn

__all__ = [
    'GraphUpsample'
]


class GraphUpsample(nn.Module):
    def __init__(self, graph):
        super(GraphUpsample, self).__init__()
        self.graph = graph

    def forward(self, x):
        N, C, T, V = x.size()

        out = x.new_empty(N, C, T, sum(len(p) for p in self.graph.part))
        for i, p in enumerate(self.graph.part):
            out[..., p] = x[..., i:i + 1]
        return out
