from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Iterator, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from abc import ABC, abstractmethod

__all__ = [
    'mgnet_resnet',
    'MGNet'
]


MODEL_URLS = {
    'mgnet': 'https://download.pytorch.org/models/'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MgNetBaseBlock(nn.Module):
    pass


class MgNetBlock(nn.Module):
    def __init__(self, n_chan: int) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=n_chan)
        self.conv1 = nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, bias=True)  # A
        self.conv2 = nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, bias=True)  # B
        self.conv3 = nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, bias=True, stride=2)  # R
        self.conv4 = nn.ConvTranspose2d(n_chan, n_chan, kernel_size=3, padding=1, bias=False, stride=2)  # P
        self.conv5 = nn.Conv2d(n_chan, n_chan, kernel_size=2, padding=0, bias=False)  # \Pi

    def forward(self, u: Tensor, f: Tensor, eta: Iterator[int]) -> Tuple[Tensor, Tensor, Iterator[int]]:
        out = u
        for i in range(next(eta)):
            out = f - self.conv1(out)
            out = torch.relu(self.batch_norm(out))
            out = out + torch.relu(self.conv2(out))
        u_l_1 = self.batch_norm(self.conv5(out))
        f = self.batch_norm(self.conv3(f - self.conv1(out)) + self.conv3(self.conv1(self.conv4(u_l_1))))
        return u_l_1, f, eta


class Bottleneck(nn.Module):
    """
    Bottleneck in torchvision places the stride for down-sampling at 3x3 convolution (self.conv2)
    while original implementation places the stride at the first 1x1 convolution (self.conv1)
    This variant improves the accuracy of the MGNet as it does for ResNet V1.5
    """
    def __init__(self):
        super(Bottleneck, self).__init__()
        pass


class TupleSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MGNet(nn.Module):
    def __init__(self, block: MgNetBaseBlock, n_iter: List[int], num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None) -> None:
        super(MGNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f'replace_stride_with_dilation should be "None" or a 3-element tuple! '
                             f'Got {replace_stride_with_dilation}')
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block)
        self.numb_blocks = len(n_iter)
        self.iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.mgblocks = TupleSequential(*(self.numb_blocks * [MgNetBlock(self.inplanes)]))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        u0 = torch.zeros(x.shape)
        etas = iter(self.iter)
        x, f, etas = self.mgblocks(u0, x, etas)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mgnet(arch: str, block, layers: List[int], pretained: bool,  progress: bool, **kwargs):
    model = MGNet(block, layers, **kwargs)
    if pretained:
        state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mgnet_resnet(pretrained=False, progress=True, **kwargs):
    return _mgnet('mgnet_resnet', MgNetBlock, [], pretrained, progress, **kwargs)


# def mgnet_iresnet(pretrained=False, progress=True, **kwargs):
