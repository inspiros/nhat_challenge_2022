import torch
import torch.nn as nn

__all__ = [
    'mpjpe',
    'MPJPELoss',
    'kld_loss',
    'KLDLoss',
]


def mpjpe(output, target, dim=1):
    """
    Mean per-joint position error (i.e. mean Euclidean distance).
    """
    return torch.mean(torch.norm(output - target, dim=dim))


def kld_loss(mu, logvar):
    """
    Kullback-Leibler Divergence Loss for VAEs.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class MPJPELoss(nn.Module):
    def __init__(self, dim):
        super(MPJPELoss, self).__init__()
        self.dim = dim

    def forward(self, output, target):
        return mpjpe(output, target, dim=self.dim)


class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return kld_loss(mu, logvar)
