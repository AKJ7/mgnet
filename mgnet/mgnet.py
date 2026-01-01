from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Iterator, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from abc import ABC, abstractmethod
import logging

__all__ = [ 'mgnet_resnet', 'MGNet' ]
logger = logging.getLogger(__name__)

MODEL_URLS = {
    'mgnet': 'https://download.pytorch.org/models/'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MGBlockSmoothing(nn.Module):
    def __init__(self, A: nn.Conv2d, B: nn.Conv2d):
        super().__init__()
        self.A = A
        self.B = B
        self.batch_normA = nn.BatchNorm2d(num_features=A.weight.size(0))
        self.batch_normB = nn.BatchNorm2d(num_features=B.weight.size(0))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, _, f = x
        out = f - self.A(u)
        out = torch.relu(self.batch_normA(out))
        out = u + torch.relu(self.batch_normB(self.B(out)))
        out = (out, out, f)
        return out


class MGBlockProlongation(nn.Module):
    def __init__(self, n_chan: int):
        super().__init__()
        self.pi = nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, stride=2, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chan)

    def forward(self, out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, u_old, f = out
        u_l_1 = torch.relu(self.batch_norm(self.pi(u)))
        out = (u_l_1, u_old, f)
        return out


class MGBlockRestriction(nn.Module):
    def __init__(self, A_old: nn.Conv2d, n_chan_u: int, n_chan_f: int):
        super().__init__()
        self.A_old = A_old
        self.R = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=2, bias=False)  # R
        self.A_l = nn.Conv2d(n_chan_u, n_chan_f, kernel_size=3, padding=1, stride=1, bias=False) # A_{l+1}
        self.batch_norm = nn.BatchNorm2d(self.A_old.weight.size(0))

    def forward(self, out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_l_1, u_l, f = out
        f = torch.relu(self.batch_norm(self.R(f - self.A_old(u_l)) + self.A_l(u_l_1)))
        out = (u_l_1, u_l, f)
        return out


class MGNet(nn.Module):
    def __init__(self, block: MgNetBaseBlock, n_iter: List[int], n_chan_u: int, n_chan_f: int, in_channels: int, num_classes=1000, in_chanel: int = 3, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer: Optional[torch.nn.Module] = None) -> None:
        super(MGNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.n_chan_u = n_chan_u
        self.n_chan_f = n_chan_f
        self.conv1 = nn.Conv2d(in_channels, self.n_chan_f, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.n_chan_f)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.numb_blocks = len(n_iter)
        self.iter = n_iter
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_chan_u, num_classes)

        layers = []
        A = nn.Conv2d(self.n_chan_u, self.n_chan_f, kernel_size=3, stride=1, padding=1, bias=False)
        for l, smooth_count in enumerate(n_iter):
            layer, A_l_1 = self._make_layer(self.n_chan_u, self.n_chan_f, smooth_count=smooth_count, A=A)
            A = A_l_1
            layers.append(layer)
        self.mg_blocks = nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        u0 = torch.zeros(f.shape)
        logger.info(f'Device: {u0.device=}')
        out = (u0, u0, f)
        out = self.mg_blocks(out)
        u, u_old, f = out
        u = self.avgpool(u)
        u = torch.flatten(u, 1)
        u = self.fc(u)
        return u

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    @staticmethod
    def _make_layer(n_chan_u: int, n_chan_f: int, smooth_count: int, A: nn.Conv2d) -> Tuple[nn.Sequential, nn.Conv2d]:
        layer = []
        for i in range(smooth_count):
            B = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=1, bias=False)  # B
            block = MGBlockSmoothing(A, B)
            layer.append(block)
        prolongation_block = MGBlockProlongation(n_chan_u)
        restriction_block = MGBlockRestriction(A, n_chan_f, n_chan_u)
        layer.append(prolongation_block)
        layer.append(restriction_block)
        seq_layer = nn.Sequential(*layer)
        return seq_layer, restriction_block.A_l


def _mgnet(arch: str, block, layers: List[int], pretrained: bool,  progress: bool, **kwargs):
    model = MGNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def mgnet_resnet(pretrained=False, progress=True, **kwargs):
    return _mgnet('mgnet_resnet', MgNetBlock, [2, 2, 2, 2, 2], pretrained, progress, **kwargs)


# def mgnet_iresnet(pretrained=False, progress=True, **kwargs):
