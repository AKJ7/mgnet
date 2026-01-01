import enum
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.datasets as torch_datasets
from pathlib import Path
import numpy as np
from enum import Enum
from typing import List


class DataSetSupported(Enum):
    CIFAR10 = torch_datasets.CIFAR10
    CIFAR100 = torch_datasets.CIFAR100
    MNIST = torch_datasets.MNIST

    def to_str(self):
        return str(self.name).lower()

    def numb_classes(self):
        return 100 if self == DataSetSupported.CIFAR100 else 10

    @classmethod
    def all(cls) -> List[str]:
        return [val.to_str() for val in cls]


class MGNetDataset(Dataset):

    def __init__(self, name: str, root: Path, train: bool, download: bool, transforms=None):
        self._name = name
        self._root = root
        self._train = train
        self._transforms = transforms
        dataset_type = DataSetSupported[name.upper()]
        self._numb_classes = dataset_type.numb_classes()
        dataset = dataset_type.value
        self._dataset = dataset(root=root, train=train, download=download, transform=transforms)

    @property
    def name(self) -> str:
        return self._name

    @property
    def classes(self) -> List[str]:
        return self.dataset.classes

    @property
    def n_classes(self) -> int:
        return len(self.dataset.classes)

    @property
    def dataset(self):
        return self._dataset

    @property
    def dim(self) :
        return self.dataset.data.shape

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item) -> Tensor:
        return self._dataset[item]

