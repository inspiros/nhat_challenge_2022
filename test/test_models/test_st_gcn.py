import torch
from models.st_gcn import CaiSTGCN


def main():
    model = CaiSTGCN(in_channels=2,
                     out_channels=2,
                     seq_len=1,
                     layout='h36m',
                     strategy='spatial',
                     dropout=0.05)

    x = torch.rand(1, 2, 1, 17)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
