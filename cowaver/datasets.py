import os
import re
import torch
from torch.nn import Module
from torch.utils.data import Dataset
from .transforms import RandomPosition, RandomAlign
from .utils import cargar_audio, extract_mel, make_image, normalizar_texto

class TinySpeakDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        transform: Module | None = None,
        classes: list[str] | None = None,
        speakers: list[str] | None = None,
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
        if speakers is None:
            speakers = self.collect_speakers(base_dir, classes)
        self.speaker_names = list(speakers)
        self.speaker_to_idx = {speaker: i for i, speaker in enumerate(self.speaker_names)}
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(base_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                root, ext = os.path.splitext(fname)
                if ext == '.wav' and not fname.startswith("."):
                    speaker_name = self.speaker_from_filename(fname)
                    if speaker_name not in self.speaker_to_idx:
                        raise ValueError(f"speaker '{speaker_name}' not found in speakers.")
                    self.samples.append((
                        os.path.join(cls_dir, root),
                        self.class_to_idx[cls],
                        speaker_name,
                        self.speaker_to_idx[speaker_name],
                    ))

    @staticmethod
    def speaker_from_filename(filename: str) -> str:
        root = os.path.splitext(os.path.basename(filename))[0]
        speaker = root.split("-", 1)[0]
        return re.sub(r"\d+$", "", speaker) or speaker

    @classmethod
    def collect_speakers(cls, base_dir: str, classes: list[str] | None = None) -> list[str]:
        if classes is None:
            classes = [
                d for d in sorted(os.listdir(base_dir))
                if not d.startswith(".") and os.path.isdir(os.path.join(base_dir, d))
            ]

        speakers = set()
        for word in classes:
            cls_dir = os.path.join(base_dir, word)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                root, ext = os.path.splitext(fname)
                if ext == ".wav" and not fname.startswith("."):
                    speakers.add(cls.speaker_from_filename(root))
        return sorted(speakers)

    @classmethod
    def collect_speakers_from_splits(cls, base_dir: str, classes: list[str] | None = None) -> list[str]:
        speakers = set()
        for split in ("train", "test"):
            split_dir = os.path.join(base_dir, split)
            if os.path.isdir(split_dir):
                speakers.update(cls.collect_speakers(split_dir, classes))
        return sorted(speakers)

    @property
    def classes(self):
        return {v: k for k, v in self.class_to_idx.items()}

    @property
    def speakers(self):
        return {v: k for k, v in self.speaker_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, target, speaker_name, speaker_id = self.samples[index]
        audio_path = file_path + ".wav"
        waveform = cargar_audio(audio_path)
        waveform = self.transform(waveform)
        return waveform, target, speaker_name, speaker_id

class ImageMelDataset(Dataset):
    def __init__(self, base_dataset: TinySpeakDataset, char_to_idx: dict[str, int], position: RandomPosition | None = None, mel_bins: int = 40, task_id: int = 1):
        self.base_dataset = base_dataset
        self.position = position
        self.mel_bins = mel_bins
        self.task_id = task_id
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.base_dataset)

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def speakers(self):
        return self.base_dataset.speakers

    def __getitem__(self, index):
        waveform, target, speaker_name, speaker_id = self.base_dataset[index]
        mel = extract_mel(waveform, mel_bins = self.mel_bins)
        word = self.classes[target]
        if self.position is None:
            x_stride, y_stride = 0, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)

        ctc_target = torch.tensor(
            [self.char_to_idx[char] for char in normalizar_texto(word)],
            dtype=torch.long,
        )
        return (image, mel), target, self.task_id, speaker_id, speaker_name, ctc_target


class RelabeledDataset(Dataset):
    """Wrap a dataset and shift its labels by a fixed offset.

    This is useful when concatenating datasets that each start their class IDs
    at zero, so labels from different sources do not collide.
    """
    def __init__(self, dataset: Dataset, label_offset: int, speaker_to_idx: dict[str, int] | None = None):
        self.dataset = dataset
        self.label_offset = label_offset
        self.speaker_to_idx = speaker_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 4:
            elements, label, task_id, ctc_target = item
            return elements, label + self.label_offset, task_id, ctc_target

        elements, label, task_id, speaker_id, speaker_name, ctc_target = item
        if self.speaker_to_idx is not None:
            speaker_id = self.speaker_to_idx[speaker_name]
        return elements, label + self.label_offset, task_id, speaker_id, speaker_name, ctc_target
