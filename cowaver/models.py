from abc import ABC, abstractmethod
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from torch.utils.data import DataLoader, Dataset, default_collate
from dataclasses import dataclass, field

class DataModule(ABC):
    def __init__(self, batch_size: int, train_set: Dataset, val_set: Dataset, test_set: Dataset):
        self.batch_size = batch_size
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    @property
    @abstractmethod
    def classes(self):
        pass

    @abstractmethod
    def elements_from_batch(self, batch) -> Tensor:
        pass

    @abstractmethod
    def labels_from_batch(self, batch):
        pass
        
    def collate_fn(self, batch):
        prefix = [item[:-1] for item in batch]
        ctc_targets = [item[-1] for item in batch]
        collated = default_collate(prefix)
        target_lengths = torch.tensor(
            [target.numel() for target in ctc_targets],
            dtype=torch.long,
        )
        return (*collated, torch.cat(ctc_targets), target_lengths)

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def inference_loader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn
        )

@dataclass
class TestResults:
    top1: float
    top3: float
    top5: float

    def __str__(self) -> str:
        return (
            f"Top-1: {self.top1 * 100:.2f}% | "
            f"Top-3: {self.top3 * 100:.2f}% | "
            f"Top-5: {self.top5 * 100:.2f}%"
        )

class TrainableModule(nn.Module):
    """A module meant to be trained."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def training_step(self, batch, batch_idx, phase: int):
        """Trains a batch.

        Parameters
        ----------
        batch : 
            The batch.
        batch_idx : i
            The index of the batch.
        phase : int
            Indicates the curent phase number.

        Returns
        -------
        out : Tensor
            The training loss.
        """
        pass

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        return TestResults(top1=0, top3=0, top5=0)

    def inference_step(self, batch: tuple) -> Tensor:
        pass

    def optimizer(self, phase: int) -> Optimizer:
        pass

    def scheduler(self, optimizer: Optimizer, phase: int) -> LRScheduler:
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    def from_checkpoint(self):
        from .checkpoints import cargar_checkpoint
        cargar_checkpoint(self, silent=True)
        return self

@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    
    @property
    def num_epochs(self) -> int:
        return len(self.train_losses)
