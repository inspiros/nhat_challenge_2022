import torch
from models.simple_gcn import SimpleGCN


def main():
    model = SimpleGCN(in_channels=2,
                      out_channels=2,
                      layout='h36m',
                      strategy='spatial',
                      dropout=0.05)

    x = torch.rand(1, 2, 1, 17)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
