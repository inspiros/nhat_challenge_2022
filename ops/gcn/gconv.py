import math

import torch
import torch.nn as nn

from ..partial_conv import PartialConv2d

__all__ = [
    'GConv',
    'PartialGConv',
]


# noinspection DuplicatedCode
class GConv(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
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
                 bias=True):
        super(GConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.linear_weight = nn.Parameter(
            torch.randn(in_channels, out_channels * kernel_size), requires_grad=True)
        if bias:
            self.linear_bias = nn.Parameter(
                torch.zeros(out_channels * kernel_size), requires_grad=True)
        else:
            self.register_parameter('linear_bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear_weight, 0, math.sqrt(
            0.5 / (self.out_channels * self.kernel_size)))
        if self.linear_bias is not None:
            nn.init.zeros_(self.linear_bias)

    def forward(self, x, A):
        """
        Notations:
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

        Args:
            x: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
            A: Input graph adjacency matrix in :math:`(K, V, V)` format

        Returns:
            x: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
            A: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        """
        assert A.size(0) == self.kernel_size

        x = torch.einsum('nctw,cd->ndtw', x, self.linear_weight)
        if self.linear_bias is not None:
            x = x + self.linear_bias.view(1, -1, 1, 1)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)

        return x, A


# noinspection DuplicatedCode
class PartialGConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 multi_channel=True,
                 return_mask=True,
                 eps=1e-8):
        super(PartialGConv, self).__init__()

        self.multi_channel = multi_channel
        self.return_mask = return_mask
        self.eps = eps

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.linear_weight = nn.Parameter(
            torch.randn(in_channels, out_channels * kernel_size), requires_grad=True)
        self.register_buffer(
            'linear_mask_update_weight',
            torch.ones(in_channels, out_channels * kernel_size)
        )
        if bias:
            self.linear_bias = nn.Parameter(
                torch.zeros(out_channels * kernel_size), requires_grad=True)
        else:
            self.register_parameter('linear_bias', None)

    def forward(self, x, A, mask=None):
        """
        Notations:
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

        Args:
            x: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
            A: Input graph adjacency matrix in :math:`(K, V, V)` format
            mask: Input binary mask matrix in :math:`(1, 1, T_{in}, V)` format

        Returns:
            x: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
            A: Graph adjacency matrix for output data in :math:`(K, V, V)` format
            mask: Output binary mask matrix in :math:`(1, 1, T_{out}, V)` format
        """
        assert A.size(0) == self.kernel_size

        x = torch.einsum('nctw,cd->ndtw', x, self.linear_weight)
        with torch.no_grad():
            mask = torch.einsum('nctw,cd->ndtw', mask, self.linear_mask_update_weight)
            mask = mask / self.in_channels
            mask_ratio = 1 / (mask + self.eps)
            mask = mask.bool().to(x.dtype)
            mask_ratio = mask_ratio * mask
        if self.linear_bias is not None:
            bias_view = self.linear_bias.view(1, -1, 1, 1)
            x = mask * (x * mask_ratio + bias_view)
        else:
            x = x * mask_ratio

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        if self.multi_channel:
            mask = mask.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        else:
            mask = mask.unsqueeze(1)

        x = torch.einsum('nkctv,kvw->nctw', x, A)
        with torch.no_grad():
            mask = torch.einsum('nkctv,kvw->nctw', mask, A)
            mask_ratio = 1 / (mask + self.eps)
            mask = mask.bool().to(x.dtype)
            mask_ratio = mask_ratio * mask

        x = x * mask_ratio

        if self.return_mask:
            return x, A, mask
        return x, A
