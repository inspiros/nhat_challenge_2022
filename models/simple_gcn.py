import torch
import torch.nn as nn

from ops.gcn import *

__all__ = ['SimpleGCN']


class GCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 temporal_kernel_size=1,
                 temporal_stride=1,
                 temporal_dilation=1,
                 dropout=0.05,
                 residual=True,
                 return_mask=False):
        super(GCNBlock, self).__init__()
        if temporal_kernel_size % 2 != 1:
            raise ValueError('temporal_kernel_size must be odd.')

        self.return_mask = return_mask

        self.gcn = GConv(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=(temporal_kernel_size, 1),
                      stride=(temporal_stride, 1),
                      dilation=(temporal_dilation, 1),
                      padding=((temporal_kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = Zero()
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=(1, 1),
                          stride=(temporal_stride, 1),
                          dilation=(temporal_dilation, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class SimpleGCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 layout='h36m',
                 strategy='spatial',
                 dropout=0.05):
        super(SimpleGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
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

        # build networks
        kernel_size = self.A.size(0)
        kernel_size_pool = self.A_pool.size(0)

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, 0.1)

        self.gcn0 = GCNBlock(self.in_channels, 32, kernel_size, dropout=dropout)
        self.gcn1 = GCNBlock(32, 64, kernel_size, dropout=dropout)
        self.pool1 = GraphMaxPool(self.graph)

        self.gcn2 = GCNBlock(64, 128, kernel_size_pool, dropout=dropout)
        self.pool2 = GraphMaxPool(self.graph_pool)

        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=(0, 0)),
            # nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.upsample1 = GraphUpsample(self.graph_pool)
        self.gcn3 = GCNBlock(128, 64, kernel_size_pool, dropout=dropout)

        self.upsample2 = GraphUpsample(self.graph)
        self.gcn4 = GCNBlock(64, 32, kernel_size, dropout=dropout)

        self.fcn = nn.Conv2d(32, self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        N, C, T, V = x.size()

        # data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        x, _ = self.gcn0(x, self.A)
        x, _ = self.gcn1(x, self.A)
        x_skip1 = x
        x = self.pool1(x)

        x, _ = self.gcn2(x, self.A_pool)
        x_skip2 = x
        x = self.pool2(x)

        x = self.conv(x)

        x = self.upsample1(x) + x_skip2
        x, _ = self.gcn3(x, self.A_pool)

        x = self.upsample2(x) + x_skip1
        x, _ = self.gcn4(x, self.A)
        x = self.fcn(x)

        return x
