import torch
from .models import DataModule
from .datasets import  ImageMelDataset

class TinyMel(DataModule):
    def __init__(self, train_set: ImageMelDataset, test_set: ImageMelDataset, batch_size: int = 32):
        full_set = train_set
        train_size = int(len(full_set) * 0.8)
        val_size = len(full_set) - train_size

        generator = torch.Generator().manual_seed(42)

        train_set, val_set = torch.utils.data.random_split(
            full_set,
            [train_size, val_size],
            generator=generator
        )
        
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
