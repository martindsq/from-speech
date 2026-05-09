import torch
from torch.nn import Module
from .models import DataModule
from .datasets import ImageMelDataset, TinySpeakDataset
from .transforms import RandomAlign, RandomPosition

class TinyMel(DataModule):
    """Build image/mel loaders for a TinySpeak-style dataset.

    Parameters
    ----------
    base_dir: str
        Dataset root containing `train` and `val` class folders.
    transform:
        Waveform transform used only for the training split. Defaults to None
        (centered).
    position: RandomPosition
        Position of stimuli in image used only for the training split. Defaults
        to None (centered).
    """
    def __init__(self, base_dir: str, transform: Module | None = None, position: RandomPosition | None = None):
        batch_size = 32
        train_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train",
                transform=transform
            ),
            position=position
        )
        val_full_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "train"
            )
        )
        test_set = ImageMelDataset(
            base_dataset=TinySpeakDataset(
                base_dir / "val"
            )
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
