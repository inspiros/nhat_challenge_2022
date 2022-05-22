import torch.nn as nn

__all__ = ['BiLSTM']


class BiLSTM(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_keypoints=17,
                 hidden_size=256,
                 num_layers=3,
                 dropout=0.05,
                 bidrectional=True):
        super(BiLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_keypoints = num_keypoints
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidrectional = bidrectional

        self.rnn = nn.LSTM(
            input_size=self.in_channels * self.num_keypoints,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidrectional,
            batch_first=True,
        )

        self.fc = nn.Linear(self.hidden_size if not self.bidrectional else self.hidden_size * 2,
                            self.out_channels * self.num_keypoints)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(N, T, C * V)

        x, _ = self.rnn(x)
        x = self.fc(x)

        x = x.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()
        return x
