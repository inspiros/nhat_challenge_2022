import torch
import torch.nn as nn

__all__ = ['VAE']


class VAE(nn.Module):
    """
    VAE model used in https://www.researchgate.net/publication/330693927_Filling_the_Gaps_Predicting_Missing_Joints_of_Human_Poses_Using_Denoising_Autoencoders.
    """

    def __init__(self, in_channels, out_channels, num_joints, latent_dim=20):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_channels * num_joints, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # hidden => mu
        self.fc_mu = nn.Linear(128, self.latent_dim)

        # hidden => logvar
        self.fc_var = nn.Linear(128, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels * num_joints),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        return mu, log_var

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        N, C, T, V = x.size()
        assert T == 1
        x = x.view(N, C * V)

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)

        out = out.view(N, -1, T, V)
        return out, mu, log_var
