"""
MGNet model in PyTorch.

This code is based on
[1] `MgNet: A Unified Framework of Multigrid and Convolutional Neural Network` by Juncai He, Jinchao Xu
[2] `Iterative Solutions of Large Sparse Systems of Equations` by Wolfgang Hackbusch

LICENSE:
    MIT License
    Copyright (c) 2022 AKJ7
"""

from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
import logging

__all__ = ('mgnet', 'MGNet')

MODEL_URLS = {'mgnet': 'https://download.pytorch.org/models/'}
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    Create a convolutional 3x3 kernel without any padding
    :param in_planes: Input channel count
    :param out_planes: Output channel count
    :param stride: Horizontal shift
    :param groups: Groups
    :param dilation: Dilation
    :return:
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    Create a convolutional 1x1 kernel without any padding
    :param in_planes: Input channel count
    :param out_planes: Output channel count
    :param stride: Horizontal shift
    :param groups: Groups
    :param dilation: Dilation
    :return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MGBlockJacobiSmoother(nn.Module):
    """
    Smoother based on the Jacobi algorithm.

    Refer to [2]{Chapter 3.2.2} or [1]{6}
    """

    def __init__(self, A: nn.Conv2d, n_chan_f: int, n_chan_u: int, index: int = 0):
        super().__init__()
        self.A = A
        self.B = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=1, bias=False)  # B
        self.batch_normA = nn.BatchNorm2d(num_features=A.weight.size(0))
        self.batch_normB = nn.BatchNorm2d(num_features=self.B.weight.size(0))

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, _, f = x
        out = f - self.A(u)
        out = torch.relu(self.batch_normA(out))
        out = u + torch.relu(self.batch_normB(self.B(out)))
        out = (out, out, f)
        return out


class MGBlockDampedJacobiSmoother(MGBlockJacobiSmoother):
    """
    Smoother based on the Damped Jacobi algorithm.

    The acceleration is for a simple 2-level smoother optimal
    only when it is set to 1.

    Technically, the acceleration should be restrained between 0 and 1. This smoother
    lets however, the training phase decide what the proper value of the acceleration
    parameter should be.

    See: [2]{Chapter 5.2 and 5.2.2}
    """

    def __init__(self, A: nn.Conv2d, n_chan_f: int, n_chan_u: int, index: int = 0):
        super().__init__(A, n_chan_f=n_chan_f, n_chan_u=n_chan_u, index=index)
        self.acceleration = nn.Parameter(torch.ones((1, 1)))

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, _, f = x
        out = f - self.A(u)
        out = torch.relu(self.batch_normA(out))
        out = u + torch.relu(self.batch_normB(self.acceleration.sigmoid() * self.B(out)))
        out = (out, out, f)
        return out


class MGMultiStepLayer(nn.ModuleDict):
    """
    Single Semi-iterative block
    """

    def __init__(self, A: nn.Conv2d, batch_normA: nn.BatchNorm2d, n_chan_f: int, n_chan_u: int):
        super().__init__()
        self.A = A
        self.bach_normA = batch_normA
        self.B_i_j = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=1, bias=False)  # B
        self.batch_normB = nn.BatchNorm2d(num_features=self.B_i_j.weight.size(0))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        u, _, f = x
        out = f - self.A(u)
        out = torch.relu(self.batch_normA(out))
        out = u + torch.relu(self.batch_normB(self.B_i_j(out)))
        return out


class MGBlockMultiStepSmoother(nn.ModuleDict):
    """
    Smoother based on the Jacobi algorithm.

    Refer to [2]{Chapter 8.2.1} or [1]{6}
    """

    def __init__(self, A: nn.Conv2d, n_chan_f: int, n_chan_u: int, index: int):
        super().__init__()
        batch_normA = nn.BatchNorm2d(num_features=A.weight.size(0))
        n_layer = index + 1
        for i in range(n_layer):
            self.add_module(
                f'multistep_block_{i}', MGMultiStepLayer(A, batch_normA, n_chan_f=n_chan_f, n_chan_u=n_chan_u)
            )
        self.coef = nn.Parameter(torch.rand(n_layer))

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor | List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor | List[Tensor], torch.Tensor]:
        u, u_old, f = x
        if isinstance(u_old, torch.Tensor):
            u_old = [u]
        target_device = next(self.parameters()).device
        out_i = torch.zeros(u.shape, device=target_device)
        softmax = nn.Softmax(dim=0)
        self.coef = nn.Parameter(softmax(self.coef))
        for index, (layer, u_l) in enumerate(zip(self.modules(), u_old)):
            out = (u_l, u_l, f)
            out_i += self.coef[index] * layer(out)
        u_old.append(out_i)
        out = (out_i, u_old, f)
        return out


class MGBlockChebyshevSmoother(nn.ModuleDict):
    """
    Smoother based on the Jacobi algorithm.

    Refer to [2]{Chapter 8.3.3} or [1]{6}
    """

    def __init__(self, A: nn.Conv2d, n_chan_f: int, n_chan_u: int, index: int = 0):
        super().__init__()
        self.A = A
        self.B = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_normA = nn.BatchNorm2d(num_features=A.weight.size(0))
        self.batch_normB = nn.BatchNorm2d(num_features=self.B.weight.size(0))
        self.w = nn.Parameter(torch.Tensor(1))

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor | List[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor | List[Tensor], torch.Tensor]:
        u, u_old, f = x
        out = f - self.A(u)
        out = torch.relu(self.batch_normA(out))
        out = u + self.w * (torch.relu(self.batch_normB(self.B(out)))) + (1 - self.w) * u_old
        out = (out, u, f)
        return out


class MGBlockProlongation(nn.Module):
    """
    Prolongation operator.

    Refer to [2]{Chapter 11.1.3} and [1]{Chapter 6}
    """

    def __init__(self, n_chan: int):
        super().__init__()
        self.pi = nn.Conv2d(n_chan, n_chan, kernel_size=3, padding=1, stride=2, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chan)

    def forward(
        self, out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, u_old, f = out
        u_l_1 = torch.relu(self.batch_norm(self.pi(u)))
        out = (u_l_1, u, f)
        return out


class MGBlockRestriction(nn.Module):
    """
    Restriction operator.

    Refer to [2]{Chapter 11.1.4} and [1]{6}
    """

    def __init__(self, A_old: nn.Conv2d, n_chan_u: int, n_chan_f: int):
        super().__init__()
        self.A_old = A_old
        self.R = nn.Conv2d(n_chan_f, n_chan_u, kernel_size=3, padding=1, stride=2, bias=False)  # R
        self.A_l = nn.Conv2d(n_chan_u, n_chan_f, kernel_size=3, padding=1, stride=1, bias=False)  # A_{l+1}
        self.batch_norm = nn.BatchNorm2d(self.A_old.weight.size(0))

    def forward(
        self, out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_l_1, u_l, f = out
        f = torch.relu(self.batch_norm(self.R(f - self.A_old(u_l)) + self.A_l(u_l_1)))
        out = (u_l_1, u_l, f)
        return out


class MGNet(nn.Module):

    SUPPORTED_SMOOTHERS = {
        'jacobi': MGBlockJacobiSmoother,
        'multistep': MGBlockMultiStepSmoother,
        'chebyshev': MGBlockChebyshevSmoother,
        'damped_jacobi': MGBlockDampedJacobiSmoother,
    }

    def __init__(
        self,
        smoother: nn.Module,
        n_iter: List[int],
        n_chan_u: int,
        n_chan_f: int,
        in_channels: int,
        n_classes: int,
        norm_layer: Optional[torch.nn.Module] = None,
    ) -> None:
        """
        Instantiate a MGNet model ... Obviously, the model should be moved onto the appropriate device.
        :param smoother: Class of the smoother to use
        :param n_iter: Number of smoothing iterations
        :param n_chan_u: Number of features channels
        :param n_chan_f: Number of data channels
        :param in_channels: Number of image channels
        :param n_classes: Number of classes
        :param norm_layer: Module to use to create normalization
        """
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
        self.fc = nn.Linear(self.n_chan_u, n_classes)
        self._smoother = smoother
        self.mg_blocks = nn.Sequential()
        A = nn.Conv2d(self.n_chan_u, self.n_chan_f, kernel_size=3, stride=1, padding=1, bias=False)
        for l, smooth_count in enumerate(n_iter):
            last_layer = l == (len(n_iter) - 1)
            layer, A_l_1 = self._make_layer(
                self.n_chan_u, self.n_chan_f, smooth_count=smooth_count, A=A, only_smooth=last_layer
            )
            A = A_l_1
            self.mg_blocks.add_module(f'layer{l}', layer)

    @property
    def parameters_count(self) -> int:
        """
        Get the number of trainable parameters in the model
        :return: Trainable parameters count
        """
        return sum(parameter.numel() for parameter in self.paramters() if parameter.required_grad)

    @property
    def loaded_device(self):
        """
        Get the device onto which the model is already loaded
        :return: Torch device used
        """
        return next(self.parameters()).device

    @staticmethod
    def supported_smoothers() -> List[str]:
        """
        Get the available Multigrid smoothers
        :return: Smoothers available
        """
        return list(MGNet.SUPPORTED_SMOOTHERS.keys())

    def _forward_impl(self, x: Tensor) -> Tensor:
        f = self.conv1(x)
        f = self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        u0 = torch.zeros(f.shape, device=self.loaded_device)
        out = (u0, u0, f)
        out = self.mg_blocks(out)
        u, u_old, f = out
        u = self.avgpool(u)
        u = torch.flatten(u, start_dim=1)
        u = self.fc(u)
        return u

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _make_layer(
        self, n_chan_u: int, n_chan_f: int, smooth_count: int, A: nn.Conv2d, only_smooth: bool = False
    ) -> Tuple[nn.Sequential, nn.Conv2d]:
        restriction_block = None
        layer = nn.Sequential()
        for i in range(smooth_count):
            block = self._smoother(A, n_chan_f=n_chan_f, n_chan_u=n_chan_u, index=i)
            layer.append(block)
        if not only_smooth:
            prolongation_block = MGBlockProlongation(n_chan_u)
            restriction_block = MGBlockRestriction(A, n_chan_f, n_chan_u)
            layer.append(prolongation_block)
            layer.append(restriction_block)
        A_l = None if restriction_block is None else restriction_block.A_l
        return layer, A_l


def _mgnet(*, arch: str, block, pretrained: bool, progress: bool, **kwargs: Any):
    model = MGNet(block, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def mgnet(smoother: str, pretrained=False, progress=True, **kwargs):
    """
    Create MGNet model from given smoother
    :param smoother: Name of the smoother from which to create the model
    :param pretrained: Fetch pretrained model from torchHub
    :param progress: Show download progress
    :param kwargs: See Keyword arguments

    :keyword Arguments
        * smoother*: Class of the smoother to use
        * n_iter*: Number of smoothing iterations
        * *n_chan_u*: Number of features channels
        * *n_chan_f*: Number of data channels
        * *in_channels*: Number of image channels
        * *n_classes*: Number of classes
        * *norm_layer*: Module to use to create normalization
    :return: Instance of the constructed MGNet class
    """
    smoother_block = MGNet.SUPPORTED_SMOOTHERS.get(smoother)
    return _mgnet(arch=smoother, block=smoother_block, pretrained=pretrained, progress=progress, **kwargs)
