import argparse

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.h36m import H36MRestorationDataset
from datasets.transforms import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data/data_2d_h36m_gt.npz')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 1000),  # frame normalize
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [T, V, C] -> [C, T, V]
        WithMaskCompose([
            # RandomMaskKeypoint(p=0.9, temporal_p=0.5),
            RandomMaskKeypointBetween(low=0, high=3),
        ]),
    ])

    test_set = H36MRestorationDataset(
        source_file_path=args.data_file,
        partition='test',
        transform=transform,
        target_transform=transform,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for batch_id, ((X, mask), Y) in enumerate(test_loader):
        # print(X.shape, Y.shape)
        print(X.shape, mask.shape)
        print(mask)


if __name__ == '__main__':
    main()
