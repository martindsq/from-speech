from abc import ABC, abstractmethod
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
    theta_max: int
    num_phases: int
    epsilon_zero: float = 3e-4
    theta: int = 60
    epsilon_theta: float = 3e-5

    def __post_init__(self):
        if self.theta_max <= 0:
            raise ValueError("theta_max must be positive.")
        if self.num_phases <= 0:
            raise ValueError("num_phases must be positive.")
        if self.theta_max % self.num_phases != 0:
            raise ValueError("theta_max must be divisible by num_phases.")
        if self.epsilon_zero <= 0:
            raise ValueError("epsilon_zero must be positive.")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        if self.theta > self.theta_max:
            raise ValueError("theta must be less than or equal to theta_max.")
        if self.epsilon_theta <= 0 or self.epsilon_theta > self.epsilon_zero:
            raise ValueError("epsilon_theta must be in the range (0, epsilon_zero].")

    def epochs_for_phase(self, phase: int) -> int:
        return self.theta_max // self.num_phases

    def epochs_before_phase(self, phase: int) -> int:
        return sum(self.epochs_for_phase(index) for index in range(1, phase))

    @property
    def total_epochs(self) -> int:
        return self.theta_max

    def decay_epochs(self) -> int:
        return self.theta

    @property
    def end_factor(self) -> float:
        return self.epsilon_theta / self.epsilon_zero

class TrainableMixin:
    @property
    def name(self) -> str:
        pass
    
    """Training interface for modules that should not inherit TrainableModule."""
    def training_step(self, batch, batch_idx, phase: int) -> Tensor:
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

    def optimizer(self, phase: int, programme: TrainProgramme) -> Optimizer:
        pass

    def scheduler(self, optimizer: Optimizer, phase: int, programme: TrainProgramme) -> LRScheduler:
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

class TrainableModule(nn.Module):
    """A module meant to be trained."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def training_step(self, batch, batch_idx, phase: int) -> Tensor:
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

    def optimizer(self, phase: int, programme: TrainProgramme) -> Optimizer:
        pass

    def scheduler(self, optimizer: Optimizer, phase: int, programme: TrainProgramme) -> LRScheduler:
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

@dataclass
class TrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    
    @property
    def num_epochs(self) -> int:
        return len(self.train_losses)
