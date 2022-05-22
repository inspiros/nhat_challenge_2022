import torch
from models.bi_lstm import BiLSTM


def main():
    model = BiLSTM(in_channels=2,
                   out_channels=2,
                   num_layers=3)

    x = torch.rand(1, 2, 9, 17)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
