import os
import pathlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset
import torchaudio
from torchaudio.models import wav2vec2_base
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
                if ext == '.opus' and not fname.startswith("."):
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
        audio_path = file_path + ".opus"
        waveform = cargar_audio(audio_path)
        waveform = self.transform(waveform)
        return waveform, target

class ImageMelDataset(Dataset):
    def __init__(self, base_dataset: TinySpeakDataset, position: RandomPosition | None = None, mel_bins: int = 80):
        self.base_dataset = base_dataset
        self.position = position
        self.mel_bins = mel_bins

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
        return (image, mel), target

class PairedImageMelDataset(Dataset):
    """Pair phonetized and spoken mels by class and within-class order."""
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
        self.phonetized_indices_by_label = self._indices_by_label(phonetized_dataset.samples)
        self.phonetized_position_by_index = self._position_by_index(
            self.phonetized_indices_by_label
        )
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

    @staticmethod
    def _position_by_index(indices_by_label):
        position_by_index = {}
        for label_indices in indices_by_label.values():
            for position, index in enumerate(label_indices):
                position_by_index[index] = position
        return position_by_index

    def __len__(self):
        return len(self.phonetized_dataset)

    @property
    def classes(self):
        return self.phonetized_dataset.classes

    def _spoken_index(self, index: int, target: int) -> int:
        candidates = self.spoken_indices_by_label[target]
        phonetized_position = self.phonetized_position_by_index[index]
        return candidates[phonetized_position % len(candidates)]

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
        elements, label = self.dataset[index]
        return elements, label + self.label_offset

class ImageVecDataset(Dataset):
    def __init__(self, base_dataset: TinySpeakDataset, wav2vec2_path: str | Path, dispositivo: torch.device = torch.device("mps"), position: RandomPosition | None = None):
        self.base_dataset = base_dataset
        self.position = position
        self.dispositivo = dispositivo

        wav2vec2 = wav2vec2_base()
        wav2vec2.load_state_dict(torch.load(wav2vec2_path, map_location=dispositivo, weights_only=True))
        self.wav2vec2 = wav2vec2.to(dispositivo).eval()

    def __len__(self):
        return len(self.base_dataset)

    @property
    def classes(self):
        return self.base_dataset.classes

    def __getitem__(self, index):
        waveform, target = self.base_dataset[index]
        B = waveform.size(0)
        T = waveform.size(1)
        with torch.no_grad():
            waveform = waveform.to(self.dispositivo)
            features, _ = self.wav2vec2.feature_extractor(waveform, length=torch.full((B,), T, dtype=torch.long, device=self.dispositivo))
        word = self.classes[target]
        if self.position is None:
            x_stride, y_stride = 0, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)
        return (image, features), target

class PairedImageVecDataset(Dataset):
    """Pair phonetized and spoken vec by class and within-class order."""
    def __init__(
        self,
        phonetized_dataset: TinySpeakDataset,
        spoken_dataset: TinySpeakDataset,
        wav2vec2_path: str | Path,
        dispositivo: torch.device = torch.device("mps"),
        position: RandomPosition | None = None,
        random_pairing: bool = False,
        seed: int = 42,
    ):
        if phonetized_dataset.classes != spoken_dataset.classes:
            raise ValueError("phonetized and spoken datasets must have the same classes.")

        self.phonetized_dataset = phonetized_dataset
        self.spoken_dataset = spoken_dataset
        self.position = position
        self.dispositivo = dispositivo
        self.random_pairing = random_pairing
        self.seed = seed
        self.phonetized_indices_by_label = self._indices_by_label(phonetized_dataset.samples)
        self.phonetized_position_by_index = self._position_by_index(
            self.phonetized_indices_by_label
        )
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

        wav2vec2 = wav2vec2_base()
        wav2vec2.load_state_dict(torch.load(wav2vec2_path, map_location=dispositivo, weights_only=True))
        self.wav2vec2 = wav2vec2.to(dispositivo).eval()

    @staticmethod
    def _indices_by_label(samples):
        indices_by_label = {}
        for index, sample in enumerate(samples):
            label = sample[1]
            indices_by_label.setdefault(label, []).append(index)
        return indices_by_label

    @staticmethod
    def _position_by_index(indices_by_label):
        position_by_index = {}
        for label_indices in indices_by_label.values():
            for position, index in enumerate(label_indices):
                position_by_index[index] = position
        return position_by_index

    def __len__(self):
        return len(self.phonetized_dataset)

    @property
    def classes(self):
        return self.phonetized_dataset.classes

    def _spoken_index(self, index: int, target: int) -> int:
        candidates = self.spoken_indices_by_label[target]
        phonetized_position = self.phonetized_position_by_index[index]
        return candidates[phonetized_position % len(candidates)]

    def __getitem__(self, index):
        phonetized_waveform, target = self.phonetized_dataset[index]
        spoken_index = self._spoken_index(index, target)
        spoken_waveform, spoken_target = self.spoken_dataset[spoken_index]
        if spoken_target != target:
            raise ValueError("paired samples must have the same target.")

        waveforms = torch.vstack([phonetized_waveform, spoken_waveform]).to(self.dispositivo)
        B = waveforms.size(0)
        T = waveforms.size(1)
        with torch.no_grad():
            features, _ = self.wav2vec2.extract_features(waveforms)

        features = features[7]
            
        word = self.classes[target]
        if self.position is None:
            x_stride, y_stride = 0, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)

        return (image, features[0:1], features[1:2]), target
