import torchvision.datasets as torch_datasets
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from PIL.Image import Image


class MGNetDataset(Dataset):

    SUPPORTED_DATASETS = {
        'cifar10': torch_datasets.CIFAR10,
        'cifar100': torch_datasets.CIFAR100,
        'mnist': torch_datasets.MNIST,
    }

    def __init__(self, name: str, root: Path, train: bool, download: bool, transforms=None):
        assert (
            name in self.supported_datasets()
        ), f'{name} dataset not supported. Possible options include: {self.supported_datasets()}'
        dataset = MGNetDataset.SUPPORTED_DATASETS[name]
        self._name = name
        self._root = root
        self._train = train
        self._dataset = dataset(root=root, train=train, download=download, transform=transforms)

    @staticmethod
    def supported_datasets() -> List[str]:
        return list(MGNetDataset.SUPPORTED_DATASETS.keys())

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
    def dim(self):
        return self.dataset.data.shape

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item) -> Tuple[Image, int]:
        return self._dataset[item]
