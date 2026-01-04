#!/usr/bin/env python3

import logging
import argparse
import torch
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List
from mgnet.dataset import MGNetDataset, DataSetSupported
from mgnet.mgnet import mgnet_resnet

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def _setup_logger(verbosity: int) -> None:
    levels = (logging.WARNING, logging.INFO, logging.DEBUG)
    log_format = '%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)-35s: %(message)s'
    assert verbosity <= len(levels), f'Array overflow! Expected at most: {len(levels)}. Got {verbosity=}'
    log_level = levels[verbosity]
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout, force=True)
    logger.debug(f'Verbosity set to: {verbosity=}')


def main(lr: float, batch_size: int, momentum: float, weight_decay: float, max_epochs: int, save_interval: int, dataset: str, verbosity: int,
         device: str, n_chan_u: int, n_chan_f: int) -> Dict[str, List[float]]:
    _setup_logger(verbosity)
    logger.info(f'Started main tests')
    logger.info(f'Learning rate: {lr}, {batch_size=}, {momentum=}, {weight_decay=}, {max_epochs=}, {dataset=}, {device=}')
    train_dataset = MGNetDataset(dataset, Path('./data'), train=True, download=True, transforms=transform_train)
    test_dataset = MGNetDataset(dataset, Path('./data'), train=False, download=True, transforms=transform_test)
    train_loader = DataLoader(train_dataset.dataset, batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset.dataset, batch_size, shuffle=False, num_workers=2)
    torch_device = torch.device(device)
    n_elements, width, height, in_channels = train_dataset.dim
    net = (mgnet_resnet(False, in_channels=in_channels, n_chan_u=n_chan_u, n_chan_f=n_chan_f)
           .to(device=torch_device))
    # logger.info(f'Model size: {net.parameters_count}')
    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []

    def train(epoch: int):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_id, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(torch_device), targets.to(torch_device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_losses.append(correct / total)
        logger.info(f'Epoch: {epoch} | Train loss: {train_loss / len(train_loader)}. Acc: {100. * correct / total}%')

    def test(epoch: int):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(torch_device), targets.to(torch_device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                test_losses.append(correct / total)
            logger.info(f'Epoch: {epoch} | Test loss: {test_loss / len(test_loader)}, Acc: {100. * correct / total}%')

    for current_epoch in range(1, max_epochs + 1):
        train(current_epoch)
        test(current_epoch)
        if current_epoch % save_interval == 0 or current_epoch == 1:
            dest = Path(f'mgnet_{current_epoch}.pth')
            torch.save(net.state_dict(), dest)
            logger.info(f'Model saved at {dest}')

    return {'train': train_losses, 'test': test_losses}

    # fig, (train_ax, test_ax) = plt.subplots(2)
    # fig.suptitle('Training results')
    # train_ax.plot(range(1, len(train_losses)), train_losses)
    # train_ax.set_title('Train loss')
    # test_ax.plot(range(1, len(test_losses)), test_losses)
    # test_ax.set_title('Test loss')
    # plt.show()


if __name__ == '__main__':
    # TODO: Add to uv: PYTHONPATH="${PYTHONPATH}:${PWD}" /home/hp/.local/bin/uv run tests/test_inference.py
    parser = argparse.ArgumentParser(description='Train and test MGNet')
    parser.add_argument('--lr', help='Learning rate', default=1e-1, type=float)
    parser.add_argument('--batch_size', help='Batch Size', default=128, type=int)
    parser.add_argument('--momentum', help='Momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', help='Weight decay', default=5e-4, type=float)
    parser.add_argument('--max_epochs', help='Max epochs', default=2, type=int)
    parser.add_argument('--save_interval', help='Interval at which the model should be saved', default=10, type=int)
    parser.add_argument('--dataset', help='Dataset to train and test upon', default=DataSetSupported.CIFAR10.to_str(), choices=DataSetSupported.all(), type=str)
    parser.add_argument('--device', help='Device to use', default=_DEFAULT_DEVICE, type=str)
    parser.add_argument('--n_chan_u', help='Number of channels to use of u', default=256, type=int)
    parser.add_argument('--n_chan_f', help='Number of channels to use of f', default=256, type=int)
    parser.add_argument('-v', '--verbosity', help='Set verbosity level', action='count', default=0)
    # args = parser.parse_args()
    args = parser.parse_args(args=['--lr', '0.1',
                                   '--batch_size', '128',
                                   '--momentum', '0.9',
                                   '--weight_decay', '5e-4',
                                   '--max_epochs', '120',
                                   '--save_interval', '10',
                                   '--dataset', 'CIFAR10'.lower(),
                                   '-vv',
                                   '--n_chan_u', '256',
                                   '--n_chan_f', '256',
                                   '--device', _DEFAULT_DEVICE])
    main(**args.__dict__)
