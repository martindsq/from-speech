import torch
from torch.nn import Module
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from .models import DataModule
from .datasets import ImageMelDataset, RelabeledDataset, TinySpeakDataset
from .transforms import RandomPosition

class TinyMel(DataModule):
    """Build image/mel loaders for a TinySpeak-style dataset.

    Parameters
    ----------
    base_dir: str
        Dataset root containing `train` and `test` class folders.
    transform:
        Waveform transform used only for the training split. Defaults to None
        (centered).
    position: RandomPosition
        Position of stimuli in image used only for the training split. Defaults
        to None (centered).
    """
    def __init__(self, base_dir: str, mel_bins: int = 40, transform: Module | None = None, position: RandomPosition | None = None):
        batch_size = 32
        train_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train",
                transform=transform
            ),
            position=position,
            mel_bins=mel_bins
        )
        val_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train"
            ),
            mel_bins=mel_bins
        )
        test_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "test"
            ),
            mel_bins=mel_bins
        )

        train_size = int(len(train_full_set) * 0.8)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(train_full_set), generator=generator).tolist()

        train_set = torch.utils.data.Subset(train_full_set, indices[:train_size])
        val_set = torch.utils.data.Subset(val_full_set, indices[train_size:])
        
        super().__init__(batch_size, train_set, val_set, test_set)
        self._mel_prototypes = None

    @property
    def classes(self):
        return self.test_set.classes

    def elements_from_batch(self, batch):
        (images, mels), _ = batch
        return (images, mels)

    def labels_from_batch(self, batch):
        _, labels = batch
        return labels

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            (_, mel0), label0 = self.test_set[0]
            mel0 = mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros((num_classes, *mel0.shape), dtype=mel0.dtype)
            counts = torch.zeros(num_classes, dtype=torch.long)

            for (_, mel), label in self.test_set:
                mel = mel.squeeze(0)
                sums[label] += mel
                counts[label] += 1

            counts = counts.clamp_min(1)
            self._mel_prototypes = sums / counts[:, None, None]

        if dispositivo is None:
            return self._mel_prototypes
        return self._mel_prototypes.to(dispositivo)


class MixedTinyMel(DataModule):
    """Compose multiple TinyMel data modules.

    `proportions` controls the expected source mix for the training loader. For
    example, `[0.7, 0.3]` samples roughly 70% from the first dataset and 30%
    from the second, independent of the source dataset sizes.
    """
    def __init__(
        self,
        datamodules: list[TinyMel],
        proportions: list[float] | None = None,
        names: list[str] | None = None,
    ):
        if len(datamodules) == 0:
            raise ValueError("MixedTinyMel needs at least one data module.")

        if proportions is not None:
            if len(proportions) != len(datamodules):
                raise ValueError("len(proportions) must be len(datamodules).")
            if any(p < 0 for p in proportions) or sum(proportions) <= 0:
                raise ValueError("proportions must be positive and sum>0.")
            if any(len(data.train_set) == 0 for data in datamodules):
                raise ValueError("all training sets must be non-empty.")

        if names is not None and len(names) != len(datamodules):
            raise ValueError("names must match the number of data modules.")

        if names is None:
            names = [f"dataset_{i}" for i in range(len(datamodules))]

        label_offsets = []
        offset = 0
        classes = {}
        for name, data in zip(names, datamodules):
            label_offsets.append(offset)
            for label, class_name in data.classes.items():
                classes[offset + label] = f"{name}:{class_name}"
            offset += len(data.classes)

        train_sets = [
            RelabeledDataset(data.train_set, label_offset)
            for data, label_offset in zip(datamodules, label_offsets)
        ]
        val_sets = [
            RelabeledDataset(data.val_set, label_offset)
            for data, label_offset in zip(datamodules, label_offsets)
        ]
        test_sets = [
            RelabeledDataset(data.test_set, label_offset)
            for data, label_offset in zip(datamodules, label_offsets)
        ]

        batch_size = 32
        self.datamodules = datamodules
        self.proportions = proportions
        self._classes = classes
        self._mel_prototypes = None

        super().__init__(
            batch_size,
            ConcatDataset(train_sets),
            ConcatDataset(val_sets),
            ConcatDataset(test_sets),
        )

    @property
    def classes(self):
        return self._classes

    def elements_from_batch(self, batch):
        (images, mels), _ = batch
        return (images, mels)

    def labels_from_batch(self, batch):
        _, labels = batch
        return labels

    def train_loader(self) -> DataLoader:
        if self.proportions is None:
            return super().train_loader()

        weights = []
        for data, proportion in zip(self.datamodules, self.proportions):
            weights.extend([proportion / len(data.train_set)] * len(data.train_set))

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(self.train_set),
            replacement=True,
        )

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            (_, mel0), _ = self.test_set[0]
            mel0 = mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros((num_classes, *mel0.shape), dtype=mel0.dtype)
            counts = torch.zeros(num_classes, dtype=torch.long)

            for (_, mel), label in self.test_set:
                mel = mel.squeeze(0)
                sums[label] += mel
                counts[label] += 1

            counts = counts.clamp_min(1)
            self._mel_prototypes = sums / counts[:, None, None]

        if dispositivo is None:
            return self._mel_prototypes
        return self._mel_prototypes.to(dispositivo)
