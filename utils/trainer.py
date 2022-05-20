import sys

import torch
from tqdm import tqdm

from models.losses import kld_loss

__all__ = [
    'train',
    'test',
    'train_vae',
    'test_vae'
]


def train(train_loader, model, criterion, optimizer, device='cpu'):
    """train function"""
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout)
    for batch_id, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        # centerize around spine
        Y_c = Y[..., 7:8]
        X = X - Y_c
        Y = Y - Y_c

        optimizer.zero_grad()
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

        # centerize around spine
        Y_c = Y[..., 7:8]
        X = X - Y_c
        Y = Y - Y_c

        Y_rec = model(X)

        metric_value = metric(Y_rec, Y)

        running_metric += metric_value.item() * X.size(0)
        pbar.set_description(f'[Validation iter {batch_id + 1}/{len(test_loader)}]'
                             f' batch_metric={metric_value.item():.03f}')
    return running_metric / len(test_loader.dataset)


def train_vae(train_loader, model, criterion, optimizer, device='cpu'):
    """train function"""
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout)
    for batch_id, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        # centerize around spine
        Y_c = Y[..., 7:8]
        X = X - Y_c
        Y = Y - Y_c

        optimizer.zero_grad()
        Y_rec, mu, log_var = model(X)
        loss = criterion(Y_rec, Y) + kld_loss(mu, log_var)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        pbar.set_description(f'[Training iter {batch_id + 1}/{len(train_loader)}]'
                             f' batch_loss={loss.item():.03f}')
    return running_loss / len(train_loader.dataset)


@torch.no_grad()
def test_vae(test_loader, model, metric, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout)
    for batch_id, (X, Y) in enumerate(pbar):
        X = X.to(device)
        Y = Y.to(device)

        # centerize around spine
        Y_c = Y[..., 7:8]
        X = X - Y_c
        Y = Y - Y_c

        Y_rec, _, _ = model(X)

        metric_value = metric(Y_rec, Y)

        running_metric += metric_value.item() * X.size(0)
        pbar.set_description(f'[Validation iter {batch_id + 1}/{len(test_loader)}]'
                             f' batch_metric={metric_value.item():.03f}')
    return running_metric / len(test_loader.dataset)
