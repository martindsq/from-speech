import os
import random
from torch.nn import Module
from torch.utils.data import Dataset
from .transforms import RandomPosition, RandomAlign
from .utils import cargar_audio, extract_mel, make_image

class TinySpeakDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform: Module | None = None,
        classes: list[str] | None = None,
    ):
        self.base_dir = base_dir
        if transform is None:
            self.transform = RandomAlign()
        else:
            self.transform = transform
        if classes is None:
            classes = [
                d for d in sorted(os.listdir(base_dir))
                if not d.startswith(".") and os.path.isdir(os.path.join(base_dir, d))
            ]
        else:
            classes = list(classes)
            missing = [
                cls for cls in classes
                if not os.path.isdir(os.path.join(base_dir, cls))
            ]
            if missing:
                raise ValueError(f"classes not found in {base_dir}: {missing}")
        self.words = classes
        self.class_to_idx = {word: i for i, word in enumerate(self.words)}
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(base_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                root, ext = os.path.splitext(fname)
                if ext == '.wav' and not fname.startswith("."):
                    self.samples.append((
                        os.path.join(cls_dir, root),
                        self.class_to_idx[cls],
                    ))

    @property
    def classes(self):
        return {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, target = self.samples[index]
        audio_path = file_path + ".wav"
        waveform = cargar_audio(audio_path)
        waveform = self.transform(waveform)
        return waveform, target

class ImageMelDataset(Dataset):
    def __init__(self, base_dataset: TinySpeakDataset, position: RandomPosition | None = None, mel_bins: int = 40, task_id: int = 1):
        self.base_dataset = base_dataset
        self.position = position
        self.mel_bins = mel_bins
        self.task_id = task_id

    def __len__(self):
        return len(self.base_dataset)

    @property
    def classes(self):
        return self.base_dataset.classes

    def __getitem__(self, index):
        waveform, target = self.base_dataset[index]
        mel = extract_mel(waveform, mel_bins = self.mel_bins)
        word = self.classes[target]
        if self.position is None:
            x_stride, y_stride = 0, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)

        return (image, mel), target, self.task_id


class PairedImageMelDataset(Dataset):
    """Pair phonetized and spoken mels by class, not by recording index."""
    def __init__(
        self,
        phonetized_dataset: TinySpeakDataset,
        spoken_dataset: TinySpeakDataset,
        position: RandomPosition | None = None,
        mel_bins: int = 40,
        random_pairing: bool = False,
        seed: int = 42,
    ):
        if phonetized_dataset.classes != spoken_dataset.classes:
            raise ValueError("phonetized and spoken datasets must have the same classes.")

        self.phonetized_dataset = phonetized_dataset
        self.spoken_dataset = spoken_dataset
        self.position = position
        self.mel_bins = mel_bins
        self.random_pairing = random_pairing
        self.seed = seed
        self.spoken_indices_by_label = self._indices_by_label(spoken_dataset.samples)
        missing_spoken_labels = [
            label
            for label in phonetized_dataset.classes
            if label not in self.spoken_indices_by_label
        ]
        if missing_spoken_labels:
            missing_classes = [
                phonetized_dataset.classes[label]
                for label in missing_spoken_labels
            ]
            raise ValueError(f"spoken dataset has no samples for classes: {missing_classes}")

    @staticmethod
    def _indices_by_label(samples):
        indices_by_label = {}
        for index, sample in enumerate(samples):
            label = sample[1]
            indices_by_label.setdefault(label, []).append(index)
        return indices_by_label

    def __len__(self):
        return len(self.phonetized_dataset)

    @property
    def classes(self):
        return self.phonetized_dataset.classes

    def _spoken_index(self, index: int, target: int) -> int:
        candidates = self.spoken_indices_by_label[target]
        if self.random_pairing:
            return random.choice(candidates)

        generator = random.Random(self.seed + index)
        return candidates[generator.randrange(len(candidates))]

    def __getitem__(self, index):
        phonetized_waveform, target = self.phonetized_dataset[index]
        spoken_index = self._spoken_index(index, target)
        spoken_waveform, spoken_target = self.spoken_dataset[spoken_index]
        if spoken_target != target:
            raise ValueError("paired samples must have the same target.")

        phonetized_mel = extract_mel(phonetized_waveform, mel_bins=self.mel_bins)
        spoken_mel = extract_mel(spoken_waveform, mel_bins=self.mel_bins)
        word = self.classes[target]
        if self.position is None:
            x_stride, y_stride = 0, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)

        return (image, phonetized_mel, spoken_mel), target


class RelabeledDataset(Dataset):
    """Wrap a dataset and shift its labels by a fixed offset.

    This is useful when concatenating datasets that each start their class IDs
    at zero, so labels from different sources do not collide.
    """
    def __init__(self, dataset: Dataset, label_offset: int):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elements, label, task_id = self.dataset[index]
        return elements, label + self.label_offset, task_id
