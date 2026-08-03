from pathlib import Path
import torch
from torch import nn, Generator
from torch.utils.data import Subset
from .models import DataModule
from .datasets import PairedImageMelDataset, TinySpeakDataset
from .transforms import RandomPosition

class TinyPairedMel(DataModule):
    """Build image/phonetized-mel/spoken-mel loaders from paired class
    datasets. The two audio datasets are paired by class label and
    within-class file order, so matching folder order can represent
    matching speakers.
    """
    def __init__(
        self,
        phonetized_dir: str | Path,
        spoken_dir: str | Path,
        mel_bins: int = 80,
        transform: nn.Module | None = None,
        position: RandomPosition | None = None,
        classes: list[str] | None = None,
        seed: int = 42,
    ):
        batch_size = 32
        phonetized_dir = Path(phonetized_dir)
        spoken_dir = Path(spoken_dir)

        train_full_set = PairedImageMelDataset(
            phonetized_dataset=TinySpeakDataset(
                phonetized_dir / "train",
                transform=transform,
                classes=classes,
            ),
            spoken_dataset=TinySpeakDataset(
                spoken_dir / "train",
                transform=transform,
                classes=classes,
            ),
            position=position,
            mel_bins=mel_bins,
        )
        val_full_set = PairedImageMelDataset(
            phonetized_dataset=TinySpeakDataset(
                phonetized_dir / "train",
                classes=classes,
            ),
            spoken_dataset=TinySpeakDataset(
                spoken_dir / "train",
                classes=classes,
            ),
            mel_bins=mel_bins,
        )
        test_set = PairedImageMelDataset(
            phonetized_dataset=TinySpeakDataset(
                phonetized_dir / "test",
                classes=classes,
            ),
            spoken_dataset=TinySpeakDataset(
                spoken_dir / "test",
                classes=classes,
            ),
            mel_bins=mel_bins,
        )

        samples = train_full_set.phonetized_dataset.samples
        generator = Generator().manual_seed(seed)
        train_indices, val_indices = TinyPairedMel._split(samples, generator)

        train_set = Subset(train_full_set, train_indices)
        val_set = Subset(val_full_set, val_indices)

        super().__init__(batch_size, train_set, val_set, test_set)
        self._mel_prototypes = None

    @property
    def classes(self):
        return self.test_set.classes

    def labels_from_batch(self, batch):
        return batch[1]
    
    def _split(samples, generator: Generator, train_ratio: float = 0.8):
        indices_by_label = {}
        for index, sample in enumerate(samples):
            label = sample[1]
            indices_by_label.setdefault(label, []).append(index)
        train_indices = []
        val_indices = []
        for label in sorted(indices_by_label):
            label_indices = indices_by_label[label]
            permutation = torch.randperm(
                len(label_indices),
                generator=generator
            ).tolist()
            shuffled = [label_indices[i] for i in permutation]

            train_size = int(round(len(shuffled) * train_ratio))
            if len(shuffled) > 1:
                train_size = min(max(train_size, 1), len(shuffled) - 1)

            train_indices.extend(shuffled[:train_size])
            val_indices.extend(shuffled[train_size:])

        return train_indices, val_indices

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            item0 = self.test_set[0]
            (_, _, spoken_mel0), _ = item0[:2]
            spoken_mel0 = spoken_mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros(
                (num_classes, *spoken_mel0.shape),
                dtype=spoken_mel0.dtype
            )
            counts = torch.zeros(num_classes, dtype=torch.long)

            for item in self.test_set:
                (_, _, spoken_mel), label = item[:2]
                spoken_mel = spoken_mel.squeeze(0)
                sums[label] += spoken_mel
                counts[label] += 1

            counts = counts.clamp_min(1)
            self._mel_prototypes = sums / counts[:, None, None]
        if dispositivo is None:
            return self._mel_prototypes
        return self._mel_prototypes.to(dispositivo)
