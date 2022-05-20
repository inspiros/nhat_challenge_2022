import torch.nn as nn

__all__ = ['AutoEncoder']


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=17, dropout=0.05):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * num_joints, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, out_channels * num_joints),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        assert T == 1
        x = x.view(N, C * V)

        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(N, -1, T, V)
        return x
