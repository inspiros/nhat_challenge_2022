import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.h36m import H36MRestorationDataset
from models.losses import *
from models.ae import AutoEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data/data_2d_h36m_gt.npz')
    parser.add_argument('--checkpoint_dir', default='checkpoints')

    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--max_epoch', default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--val_frequency', type=int, default=1,
                        help='number of epochs between each validation.')
    parser.add_argument('--save_frequency', type=int, default=5,
                        help='number of epochs between each checkpoint.')

    parser.add_argument('--resume', action='store_true',
                        help='continue from last epoch.')
    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


def train(train_loader, model, criterion, optimizer, device='cpu'):
    """train function"""
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout)
    for batch_id, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        # Y_rec, mu, logvar = model(X)
        # loss = criterion(Y_rec, Y) + 0.1 * kld_loss(mu, logvar)
        Y_rec = model(X)
        loss = criterion(Y_rec, Y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        pbar.set_description(f'[Training iter {batch_id + 1}/{len(train_loader)}]'
                             f' batch_loss={loss.item():.03f}')
    return running_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_loader, model, metric, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout)
    for batch_id, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        Y_rec = model(X)
        metric_value = metric(Y_rec, Y)

        running_metric += metric_value.item() * X.size(0)
        pbar.set_description(f'[Validation iter {batch_id + 1}/{len(test_loader)}]'
                             f' batch_metric={metric_value.item():.03f}')
    return running_metric / len(test_loader.dataset)


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x / 1000),  # frame normalize
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [T, V, C] -> [C, T, V]
    ])

    train_set = H36MRestorationDataset(
        source_file_path=args.data_file,
        partition='train',
        transform=transform,
        target_transform=transform,
    )
    test_set = H36MRestorationDataset(
        source_file_path=args.data_file,
        partition='test',
        transform=transform,
        target_transform=transform,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = AutoEncoder(in_channels=2,
                        out_channels=2,
                        num_joints=17,
                        ).to(args.device)
    criterion = MPJPELoss(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # evaluation metric
    def metric(output, target):
        output = output * 1000
        target = target * 1000
        return mpjpe(output, target, dim=1)

    # load previous states
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint_dir):
        last_file = os.path.join(args.checkpoint_dir, sorted(os.listdir(args.checkpoint_dir))[-1])
        last_state_dict = torch.load(last_file, map_location=args.device)
        start_epoch = last_state_dict['epoch'] - 1
        model.load_state_dict(last_state_dict['model_state_dict'])
        optimizer.load_state_dict(last_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(last_state_dict['scheduler_state_dict'])

    # Training Loop
    for epoch in range(start_epoch, args.max_epoch):
        print(f'[Epoch {epoch + 1} / {args.max_epoch}]')
        # train
        epoch_loss = train(train_loader, model, criterion, optimizer, args.device)
        scheduler.step()
        print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
              f'train_loss={epoch_loss:.4f}')

        # val
        if (epoch + 1) % args.val_frequency == 0 or epoch == args.max_epoch - 1:
            epoch_mpjpe = test(test_loader, model, metric, args.device)
            print(f'[Epoch {epoch + 1} / {args.max_epoch}] '
                  f'val_mpjpe={epoch_mpjpe:.4f}')

        # save
        if (epoch + 1) % args.save_frequency == 0 or epoch == args.max_epoch - 1:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(args.checkpoint_dir, f'model_{epoch}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_file)
        print()


if __name__ == '__main__':
    main()
