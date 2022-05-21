import torch

from models.ae import PartialAutoEncoder


def main():
    model = PartialAutoEncoder(in_channels=2,
                               out_channels=2,
                               num_joints=17,
                               dropout=0.05)

    x = torch.rand(1, 2, 1, 17)
    mask = torch.randint(0, 2, x.size()).type_as(x)
    y = model(x, mask)
    print(y.shape)


if __name__ == '__main__':
    main()
