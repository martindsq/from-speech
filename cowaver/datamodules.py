from pathlib import Path
import torch
from torch.nn import Module
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from .models import DataModule
from .datasets import ImageMelDataset, PairedImageMelDataset, RelabeledDataset, TinySpeakDataset
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
    def __init__(
        self,
        base_dir: str | Path,
        mel_bins: int = 40,
        transform: Module | None = None,
        position: RandomPosition | None = None,
        task_id: int = 1,
        classes: list[str] | None = None,
    ):
        batch_size = 32
        self.task_id = task_id
        train_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train",
                transform=transform,
                classes=classes,
            ),
            position=position,
            mel_bins=mel_bins,
            task_id=task_id,
        )
        val_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train",
                classes=classes,
            ),
            mel_bins=mel_bins,
            task_id=task_id,
        )
        test_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "test",
                classes=classes,
            ),
            mel_bins=mel_bins,
            task_id=task_id,
        )

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = self._stratified_split_indices(
            train_full_set.base_dataset.samples,
            train_ratio=0.8,
            generator=generator,
        )

        train_set = torch.utils.data.Subset(train_full_set, train_indices)
        val_set = torch.utils.data.Subset(val_full_set, val_indices)
        
        super().__init__(batch_size, train_set, val_set, test_set)
        self._mel_prototypes = None

    @staticmethod
    def _stratified_split_indices(samples, train_ratio: float, generator: torch.Generator):
        indices_by_label = {}
        for index, sample in enumerate(samples):
            label = sample[1]
            indices_by_label.setdefault(label, []).append(index)

        train_indices = []
        val_indices = []
        for label in sorted(indices_by_label):
            label_indices = indices_by_label[label]
            permutation = torch.randperm(len(label_indices), generator=generator).tolist()
            shuffled = [label_indices[i] for i in permutation]

            train_size = int(round(len(shuffled) * train_ratio))
            if len(shuffled) > 1:
                train_size = min(max(train_size, 1), len(shuffled) - 1)

            train_indices.extend(shuffled[:train_size])
            val_indices.extend(shuffled[train_size:])

        return train_indices, val_indices

    @property
    def classes(self):
        return self.test_set.classes

    def labels_from_batch(self, batch):
        return batch[1]

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            item0 = self.test_set[0]
            (_, mel0), label0 = item0[:2]
            mel0 = mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros((num_classes, *mel0.shape), dtype=mel0.dtype)
            counts = torch.zeros(num_classes, dtype=torch.long)

            for item in self.test_set:
                (_, mel), label = item[:2]
                mel = mel.squeeze(0)
                sums[label] += mel
                counts[label] += 1

            counts = counts.clamp_min(1)
            self._mel_prototypes = sums / counts[:, None, None]

        if dispositivo is None:
            return self._mel_prototypes
        return self._mel_prototypes.to(dispositivo)


class FilteredTinyMel(DataModule):
    """Evaluate a subset of classes against prototypes from the full source."""
    def __init__(self, source: TinyMel, classes: list[str]):
        class_to_label = {
            class_name: label
            for label, class_name in source.classes.items()
        }
        missing = [
            class_name for class_name in classes
            if class_name not in class_to_label
        ]
        if missing:
            raise ValueError(f"filtered classes not found in source: {missing}")

        test_labels = {
            class_to_label[class_name]
            for class_name in classes
        }
        indices = [
            index
            for index, sample in enumerate(source.test_set.base_dataset.samples)
            if sample[1] in test_labels
        ]
        if len(indices) == 0:
            raise ValueError("filtered test set is empty.")

        self.source = source
        super().__init__(
            source.batch_size,
            source.train_set,
            source.val_set,
            torch.utils.data.Subset(source.test_set, indices),
        )

    @property
    def classes(self):
        return self.source.classes

    def labels_from_batch(self, batch):
        return self.source.labels_from_batch(batch)

    def collate_fn(self, batch):
        return self.source.collate_fn(batch)

    def mel_prototypes(self, dispositivo=None):
        return self.source.mel_prototypes(dispositivo)


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

    def labels_from_batch(self, batch):
        return batch[1]

    def train_loader(self) -> DataLoader:
        if self.proportions is None:
            return super().train_loader()

        return self._proportional_loader(self.train_set, [len(data.train_set) for data in self.datamodules])

    def val_loader(self) -> DataLoader:
        if self.proportions is None:
            return super().val_loader()

        return self._proportional_loader(self.val_set, [len(data.val_set) for data in self.datamodules])

    def _proportional_loader(self, dataset, dataset_lengths: list[int]) -> DataLoader:
        weights = []
        for dataset_length, proportion in zip(dataset_lengths, self.proportions):
            weights.extend([proportion / dataset_length] * dataset_length)

        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(42),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            item0 = self.test_set[0]
            (_, mel0), _ = item0[:2]
            mel0 = mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros((num_classes, *mel0.shape), dtype=mel0.dtype)
            counts = torch.zeros(num_classes, dtype=torch.long)

            for item in self.test_set:
                (_, mel), label = item[:2]
                mel = mel.squeeze(0)
                sums[label] += mel
                counts[label] += 1

            counts = counts.clamp_min(1)
            self._mel_prototypes = sums / counts[:, None, None]

        if dispositivo is None:
            return self._mel_prototypes
        return self._mel_prototypes.to(dispositivo)


class TinyPairedMel(DataModule):
    """Build image/phonetized-mel/spoken-mel loaders from paired class datasets.

    The two audio datasets are paired by class label. Training samples draw a
    random spoken item from the same class each time; validation and test use a
    deterministic same-class pairing.
    """
    def __init__(
        self,
        phonetized_dir: str | Path,
        spoken_dir: str | Path,
        mel_bins: int = 40,
        transform: Module | None = None,
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
            random_pairing=True,
            seed=seed,
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
            random_pairing=False,
            seed=seed,
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
            random_pairing=False,
            seed=seed,
        )

        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = TinyMel._stratified_split_indices(
            train_full_set.phonetized_dataset.samples,
            train_ratio=0.8,
            generator=generator,
        )

        train_set = torch.utils.data.Subset(train_full_set, train_indices)
        val_set = torch.utils.data.Subset(val_full_set, val_indices)

        super().__init__(batch_size, train_set, val_set, test_set)
        self._mel_prototypes = None

    @property
    def classes(self):
        return self.test_set.classes

    def labels_from_batch(self, batch):
        return batch[1]

    def mel_prototypes(self, dispositivo=None):
        if self._mel_prototypes is None:
            item0 = self.test_set[0]
            (_, _, spoken_mel0), _ = item0[:2]
            spoken_mel0 = spoken_mel0.squeeze(0)
            num_classes = len(self.classes)
            sums = torch.zeros((num_classes, *spoken_mel0.shape), dtype=spoken_mel0.dtype)
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
