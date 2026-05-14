import os
from torch.nn import Module
from torch.utils.data import Dataset
from .transforms import RandomPosition, RandomAlign
from .utils import cargar_audio, extract_mel, make_image

class TinySpeakDataset(Dataset):
    def __init__(self, base_dir: str, transform: Module | None = None):
        self.base_dir = base_dir
        if transform is None:
            self.transform = RandomAlign()
        else:
            self.transform = transform
        classes = [
            d for d in sorted(os.listdir(base_dir))
            if not d.startswith(".") and os.path.isdir(os.path.join(base_dir, d))
        ]
        self.words = classes
        self.class_to_idx = {word: i for i, word in enumerate(self.words)}
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(base_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                root, ext = os.path.splitext(fname)
                if ext == '.wav' and not fname.startswith("."):
                    self.samples.append((os.path.join(cls_dir, root), self.class_to_idx[cls]))

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
    def __init__(self, base_dataset: TinySpeakDataset, position: RandomPosition | None = None, mel_bins: int = 40):
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
            x_stride, y_stride = 0.5, 0.5
        else:
            x_stride, y_stride = self.position()
        image = make_image(word, x_stride, y_stride)

        return (image, mel), target


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
