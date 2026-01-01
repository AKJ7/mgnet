import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopping:

    def __init__(self, patience=7, delta=0.0, filename=Path('checkpoint.pth'), log_level=logging.INFO):
        """
        Inspired by https://github.com/Bjarten/early-stopping-pytorch
        :param patience: How long to wait after last time validation loss improved.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement
        :param filename: Name of the checkpoint file name
        :param log_level: Lowest log severity level to dispatch
        """
        self._patience = patience
        self._counter = 0
        self._filename = filename
        self._best_score = None
        self._early_stop = False
        self._val_loss_min = np.Inf
        self._delta = delta
        logger.setLevel(log_level)
        logger.debug(f'Early-Stopping Object initialised at: {hex(id(self))}')

    @property
    def early_stop(self) -> bool:
        return self.early_stop

    def __call__(self, val_loss: float, epoch: int, model) -> None:
        score = val_loss
        if self._best_score is None:
            logger.debug(f'No best score. Storing initial value. Value: {val_loss}, epoch: {epoch}')
            self._best_score = score
            self.save_checkpoint(val_loss, epoch, model)
        elif self._best_score - self._delta > score:
            self._counter += 1
            logger.debug(f'Early-Stopping counter: {self._counter} out of patience: {self._patience}')
            if self._counter >= self._patience:
                self._early_stop = True
        else:
            self._best_score = score
            self._save_checkpoint(val_loss, epoch, model)
            self._counter = 0

    def _save_checkpoint(self, val_loss: float, epoch: int, model) -> None:
        # Path.with_stem only supported in pathlib version 3.9 onwards
        filename = self._filename.with_name(f'{self._filename.stem}_{epoch}').with_suffix(self._filename.suffix)
        logger.info(f'Validation loss decreased. {self._val_loss_min: .6f} -> {val_loss: .6f}. Saving model at: '
                    f'{filename}')
        torch.save(model.state_dict(), filename)


class Averager:
    def __init__(self, name: str):
        self._name = name
        self._current = 0
        self._average = 0
        self._sum = 0
        self._count = 0

    @property
    def count(self) -> float:
        return self._count

    @property
    def value(self) -> float:
        return self._sum

    def reset(self):
        self._current = 0
        self._average = 0
        self._sum = 0
        self._count = 0

    def insert(self, value: float, numb=1):
        self._current += value
        self._sum += value * numb
        self._count += 1
        self._average = self._sum / self._count

    def __repr__(self) -> str:
        return f'Averager: "{self._name}". Value: {self._sum}, Average: {self._average}, Count: {self._count}'
