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
        self.__patience = patience
        self.__counter = 0
        self.__filename = filename
        self.__best_score = None
        self.__early_stop = False
        self.__val_loss_min = np.Inf
        self.__delta = delta
        logger.setLevel(log_level)
        logger.debug(f'Early-Stopping Object initialised at: {hex(id(self))}')

    @property
    def early_stop(self) -> bool:
        return self.early_stop

    def __call__(self, val_loss: float, epoch: int, model) -> None:
        score = val_loss
        if self.__best_score is None:
            logger.debug(f'No best score. Storing initial value. Value: {val_loss}, epoch: {epoch}')
            self.__best_score = score
            self._save_checkpoint(val_loss, epoch, model)
        elif self.__best_score - self.__delta > score:
            self.__counter += 1
            logger.debug(f'Early-Stopping counter: {self.__counter} out of patience: {self.__patience}')
            if self.__counter >= self.__patience:
                self.__early_stop = True
        else:
            self.__best_score = score
            self._save_checkpoint(val_loss, epoch, model)
            self.__counter = 0

    def _save_checkpoint(self, val_loss: float, epoch: int, model) -> None:
        # Path.with_stem only supported in pathlib version 3.9 onwards
        filename = self.__filename.with_name(f'{self.__filename.stem}_{epoch}').with_suffix(self.__filename.suffix)
        logger.info(f'Validation loss decreased. {self.__val_loss_min: .6f} -> {val_loss: .6f}. Saving model at: '
                    f'{filename}')
        torch.save(model.state_dict(), filename)


class Averager:
    def __init__(self, name: str):
        self.__name = name
        self.__reset()

    @property
    def count(self) -> float:
        return self.__count

    @property
    def value(self) -> float:
        return self.__sum

    def __reset(self):
        self.__current = 0
        self.__average = 0
        self.__sum = 0
        self.__count = 0

    def __insert(self, value: float, numb=1):
        self.__current += value
        self.__sum += value * numb
        self.__count += 1
        self.__average = self.__sum / self.__count

    def __repr__(self) -> str:
        return f'Averager: "{self.__name}". Value: {self.__sum}, Average: {self.__average}, Count: {self.__count}'
