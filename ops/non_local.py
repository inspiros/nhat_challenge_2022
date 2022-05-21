import torch
from torch import nn

__all__ = [
    'NonLocal1d',
    'NonLocal2d',
    'NonLocal3d'
]


class _NonLocalNd(nn.Module):
    def __init__(self, dimension, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(_NonLocalNd, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 1:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if self.sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        """
        Args:
            x: (b, c, t, h, w)

        Returns:

        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = f.softmax(dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NonLocal1d(_NonLocalNd):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NonLocal1d, self).__init__(dimension=1,
                                         in_channels=in_channels,
                                         inter_channels=inter_channels,
                                         sub_sample=sub_sample,
                                         bn_layer=bn_layer)


class NonLocal2d(_NonLocalNd):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NonLocal2d, self).__init__(dimension=2,
                                         in_channels=in_channels,
                                         inter_channels=inter_channels,
                                         sub_sample=sub_sample,
                                         bn_layer=bn_layer)


class NonLocal3d(_NonLocalNd):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NonLocal3d, self).__init__(dimension=3,
                                         in_channels=in_channels,
                                         inter_channels=inter_channels,
                                         sub_sample=sub_sample,
                                         bn_layer=bn_layer)


if __name__ == '__main__':
    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NonLocal1d(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NonLocal2d(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NonLocal3d(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())
