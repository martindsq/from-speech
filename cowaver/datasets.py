import os, unicodedata, random
from torch import stack, randn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor
from .utils import cargar_audio, clip_waveform, extract_mel, make_image

EMNIST_MEAN = [0.485, 0.456, 0.406]
EMNIST_STD = [0.229, 0.224, 0.225]

class TinySpeakDataset(Dataset):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
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
        return waveform, target

class TinyEMNISTDataset(ImageFolder):
    def __init__(self, dataset_path):
        super().__init__(
            dataset_path,
            Compose([ToTensor(), Normalize(EMNIST_MEAN, EMNIST_STD)])
        )

class RandomStride:
    def __init__(self, mean=0.5, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self):
        x = randn(1).item() * self.std + self.mean
        y = randn(1).item() * self.std + self.mean

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        return x, y

class ImageMelDataset(Dataset):
    def __init__(self, base_dataset: TinySpeakDataset, stride: RandomStride | None = None):
        self.base_dataset = base_dataset
        self.stride = stride

    def __len__(self):
        return len(self.base_dataset)

    @property
    def classes(self):
        return self.base_dataset.classes

    def __getitem__(self, index):
        waveform, target = self.base_dataset[index]

        clipped_waveform = clip_waveform(waveform, duration=1.0)
        mel = extract_mel(clipped_waveform)
        word = self.classes[target]
        if self.stride is None:
            x_stride, y_stride = 0.5, 0.5
        else:
            x_stride, y_stride = self.stride()
        image = make_image(word, x_stride, y_stride)

        return (image, mel), target
