import torch
import torch.nn as nn

from ops.gcn import Graph, SubGraph, GConv, GraphMaxPool, GraphUpsample
from ops.non_local import NonLocal2d

__all__ = [
    'WangSTGCN',
    'CaiSTGCN'
]


class Zero(nn.Module):
    def forward(self, *args, **kwargs):
        return 0


class DataBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        N, C, T, V = x.size()
        # data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = super(DataBatchNorm, self).forward(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        return x


class STGCNBlock(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int):
        temporal_kernel_size (int, optional): Size of the temporal convolving kernel and graph convolving kernel
        temporal_stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

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
        super(STGCNBlock, self).__init__()
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


# noinspection PyPep8Naming
class WangSTGCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 graph_cfg,
                 edge_importance_weighting=False,
                 data_bn=True,
                 **kwargs):
        super(WangSTGCN, self).__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else nn.Identity()

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.block1 = STGCNBlock(in_channels, 16, kernel_size, 1, residual=False, **kwargs0)
        self.block2 = STGCNBlock(16, 32, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block3 = STGCNBlock(32, 32, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block4 = STGCNBlock(32, 64, kernel_size, temporal_kernel_size, 2, **kwargs)
        self.block5 = STGCNBlock(64, 64, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block6 = STGCNBlock(64, 128, kernel_size, temporal_kernel_size, 2, **kwargs)
        self.block7 = STGCNBlock(128, 64, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=(2, 1))
        self.block8 = STGCNBlock(64, 64, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block9 = STGCNBlock(64, 32, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=(2, 1))
        self.block10 = STGCNBlock(32, 32, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block11 = STGCNBlock(32, 16, kernel_size, temporal_kernel_size, 1, **kwargs)
        self.block12 = STGCNBlock(16, in_channels, kernel_size, temporal_kernel_size, 1, **kwargs)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.register_buffer('edge_importance', torch.ones(11))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()

        # data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        # forward
        x, _ = self.block1(x, self.A)
        x, _ = self.block2(x, self.A)
        x, _ = self.block3(x, self.A)
        res1 = x
        x, _ = self.block4(x, self.A)
        x, _ = self.block5(x, self.A)
        res2 = x
        x, _ = self.block6(x, self.A)
        x, _ = self.block7(x, self.A)
        x = self.upsample1(x)
        x += res2
        x, _ = self.block8(x, self.A)
        x, _ = self.block9(x, self.A)
        x = self.upsample2(x)
        x += res1
        x, _ = self.block10(x, self.A)
        x, _ = self.block11(x, self.A)
        x, _ = self.block12(x, self.A)

        return x


class CaiSTGCN(nn.Module):
    """
    http://openaccess.thecvf.com/content_ICCV_2019/papers/Cai_Exploiting_Spatial-Temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_ICCV_2019_paper.pdf
    """
    inter_channels = [64, 128, 256]

    fc_out = inter_channels[-1]
    fc_unit = 512

    def __init__(self,
                 in_channels,
                 out_channels,
                 seq_len=1,
                 data_bn=False,
                 cat=True,
                 layout='h36m',
                 strategy='spatial',
                 temporal_connection=True,
                 dropout=0.1):
        super(CaiSTGCN, self).__init__()
        if seq_len % 2 != 1:
            raise ValueError('seq_len must be odd')

        # load graph
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layout = layout
        self.strategy = strategy
        self.seq_len = seq_len
        self.temporal_connection = temporal_connection
        self.cat = cat

        # original graph
        self.graph = Graph(self.layout, self.strategy,
                           seq_len=self.seq_len if self.temporal_connection else 1)
        self.register_buffer(
            'A', torch.tensor(self.graph.A, dtype=torch.float32)
        )  # K, T*V, T*V

        # pooled graph
        self.graph_pool = SubGraph(self.layout, self.strategy,
                                   seq_len=self.seq_len if self.temporal_connection else 1)
        self.register_buffer(
            'A_pool', torch.tensor(self.graph_pool.A, dtype=torch.float32)
        )

        # build networks
        kernel_size = self.A.size(0)
        kernel_size_pool = self.A_pool.size(0)

        if data_bn:
            self.data_bn = DataBatchNorm(self.in_channels * self.graph.num_node_each, 0.1)
        else:
            self.data_bn = nn.Identity()

        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(self.in_channels, self.inter_channels[0], kernel_size, residual=False),
            STGCNBlock(self.inter_channels[0], self.inter_channels[1], kernel_size),
            STGCNBlock(self.inter_channels[1], self.inter_channels[2], kernel_size),
        ))
        self.pool1 = GraphMaxPool(self.graph)

        self.st_gcn_pool = nn.ModuleList((
            STGCNBlock(self.inter_channels[-1], self.fc_unit, kernel_size_pool),
            STGCNBlock(self.fc_unit, self.fc_unit, kernel_size_pool),
        ))
        self.pool2 = GraphMaxPool(self.graph_pool)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.fc_unit, self.fc_unit,
                      kernel_size=(3, 1) if self.seq_len > 1 else (1, 1),
                      padding=(1, 0) if self.seq_len > 1 else (0, 0)),
            nn.BatchNorm2d(self.fc_unit, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.upsample1 = GraphUpsample(self.graph_pool)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.fc_unit * 2 if self.cat else self.fc_unit,
                      self.fc_out,
                      kernel_size=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(self.fc_out, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.upsample2 = GraphUpsample(self.graph)
        self.non_local = NonLocal2d(in_channels=self.fc_out * 2 if self.cat else self.fc_out,
                                    sub_sample=False)

        # fcn for final layer prediction
        fc_in = self.inter_channels[-1] + self.fc_out if self.cat else self.inter_channels[-1]
        self.fcn = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Conv2d(fc_in,
                      self.out_channels,
                      kernel_size=(1, 1)),
        )

    def forward(self, x):
        N, C, T, V = x.size()

        # data normalization
        x = self.data_bn(x)
        if self.temporal_connection:
            x = x.view(N, C, 1, T * V)  # N, C, 1, (T*V)

        # forward GCN
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A)  # N, C, 1, (T*V)
        x = x.view(N, -1, T, V)  # N, C, T ,V

        # Pooling 1
        x_res1 = x
        x = self.pool1(x)  # N, C, T, 5

        if self.temporal_connection:
            x = x.view(N, -1, 1, T * len(self.graph.part))  # N, 512, 1, (T*5)
        x, _ = self.st_gcn_pool[0](x, self.A_pool)  # N, 512, 1, (T*5)
        x, _ = self.st_gcn_pool[1](x, self.A_pool)  # N, 512, 1, (T*5)
        x = x.view(N, -1, T, len(self.graph.part))  # N, C, T, 5

        # Pooling 2
        x_res2 = x
        x = self.pool2(x)  # N, 512, T, 1
        x = self.conv1(x)  # N, C, T, 1

        # Upsample 1
        x = self.upsample1(x)
        if self.cat:
            x = torch.cat((x_res2, x), dim=1)  # N, 1024, T, 5
        else:
            x = x_res2 + x
        x = self.conv2(x)  # N, C, T, 5

        # Upsample 2
        x = self.upsample2(x)
        if self.cat:
            x = torch.cat((x_res1, x), dim=1)
        else:
            x = x_res1 + x
        x = self.non_local(x)  # N, 2C, T, V
        x = self.fcn(x)  # N, 3, T, V

        # output
        return x


class LiSTGCN(nn.Module):
    """

    """
    def __init__(self):
        super(LiSTGCN, self).__init__()
        # TODO
        raise NotImplementedError
