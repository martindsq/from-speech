from abc import ABC, abstractmethod
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR
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
    def labels_from_batch(self, batch):
        pass
        
    def collate_fn(self, batch):
        return default_collate(batch)

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

@dataclass
class TrainProgramme:
    """

    Attributes
    ----------
    theta_max : int
    epsilon_zero : float
    theta : int
    epsilon_theta : float
    patience : int
    	Cuantas épocas esperar hasta finalizar el entrenamiento de forma
    	prematura (early stopping).
    """
    theta_max: int
    epsilon_zero: float = 3e-4
    theta: int = 60
    epsilon_theta: float = 3e-5
    patience: int = 3

    @property
    def lr(self) -> float:
        return self.epsilon_zero
    
    @property
    def num_epochs(self) -> int:
        return self.theta_max

    @property
    def start_factor(self) -> float:
        return 1.0
    
    @property
    def end_factor(self) -> float:
        return self.epsilon_theta / self.epsilon_zero

    @property
    def total_iters(self) -> float:
        return self.theta

class TrainableMixin:
    @property
    def name(self) -> str:
        pass
    
    def training_step(self, batch) -> Tensor:
        """Trains a batch.

        Parameters
        ----------
        batch : 
            The batch.

        Returns
        -------
        out : Tensor
            The training loss.
        """
        pass

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        return TestResults(top1=0, top3=0, top5=0)

    def optimizer(self, programme: TrainProgramme) -> Optimizer:
        pass

    def scheduler(self, optimizer: Optimizer, programme: TrainProgramme) -> LRScheduler:
        return LinearLR(
            optimizer,
            start_factor=programme.start_factor,
            end_factor=programme.end_factor,
            total_iters=programme.total_iters
        )

@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    
    @property
    def num_epochs(self) -> int:
        return len(self.train_losses)
