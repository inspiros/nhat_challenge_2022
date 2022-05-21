import torch
import torch.nn as nn

from ops.gcn import *

__all__ = ['SimpleGCN']


class GCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout=0.0):
        super(GCNBlock, self).__init__()
        self.gcn = GConv(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        x, A = self.gcn(x, A)
        x = self.relu(x)
        x = self.dropout(x)
        return x, A


class SimpleGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 layout='h36m',
                 strategy='spatial',
                 dropout=0.05):
        super(SimpleGCN, self).__init__()
        self.layout = layout
        self.strategy = strategy

        # original graph
        self.graph = Graph(self.layout, self.strategy, seq_len=1)
        self.register_buffer(
            'A', torch.tensor(self.graph.A, dtype=torch.float32)
        )  # K, T*V, T*V

        # pooled graph
        self.graph_pool = SubGraph(self.layout, self.strategy, seq_len=1)
        self.register_buffer(
            'A_pool', torch.tensor(self.graph_pool.A, dtype=torch.float32)
        )

        self.gcn1 = GCNBlock(in_channels, 64, self.A.size(0), dropout=dropout)
        self.pool1 = GraphMaxPool(self.graph)

        self.gcn2 = GCNBlock(64, 128, self.A_pool.size(0), dropout=dropout)
        self.pool2 = GraphMaxPool(self.graph_pool)

        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0)),
            # nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.upsample1 = GraphUpsample(self.graph_pool)
        self.gcn3 = GCNBlock(128, 64, self.A_pool.size(0), dropout=dropout)

        self.upsample2 = GraphUpsample(self.graph)
        self.gcn4 = GCNBlock(64, out_channels, self.A.size(0), dropout=dropout)

    def forward(self, x):
        N, C, T, V = x.size()

        x, _ = self.gcn1(x, self.A)
        x = self.pool1(x)

        x, _ = self.gcn2(x, self.A_pool)
        x = self.pool2(x)

        x = self.conv(x)

        x = self.upsample1(x)
        x, _ = self.gcn3(x, self.A_pool)

        x = self.upsample2(x)
        x, _ = self.gcn4(x, self.A)

        return x
