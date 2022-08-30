#!/usr/bin/env python3

import logging
import argparse
import torch
from torchvision import transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from mgnet.mgnet import mgnet_resnet
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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


def main(lr: float, batch_size: int, momentum: float, weight_decay: float, max_epochs: int, save_interval: int) -> None:
    logger.info(f'Started tests')
    logger.info(f'Learning rate: {lr}, Batch size: {batch_size}, Momentum: {momentum}, Weight decay: {weight_decay}')
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=2)
    net = mgnet_resnet(False).to(device=DEVICE)
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
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backwar()
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
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
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
    fig, (train_ax, test_ax) = plt.subplots(2)
    fig.suptitle('Training results')
    train_ax.plot(range(1, len(train_losses)), train_losses)
    train_ax.set_title('Train loss')
    test_ax.plot(range(1, len(test_losses)), test_losses)
    test_ax.set_title('Test loss')
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train and test MGNet')
    parser.add_argument('--lr', help='Learning rate', default=1e-1, type=float)
    parser.add_argument('--batch_size', help='Batch Size', default=128, type=int)
    parser.add_argument('--momentum', help='Momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', help='Weight decay', default=5e-4, type=float)
    parser.add_argument('--max_epochs', help='Max epochs', default=200, type=int)
    parser.add_argument('--save_interval', help='Interval at which the model should be saved', default=10, type=int)
    args = parser.parse_args()
    main(**args.__dict__)
