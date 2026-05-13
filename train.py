import argparse
from pathlib import Path
from torchvision.transforms import Compose
from cowaver.datamodules import TinyMel
from cowaver.modules import CoWaver
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red,
    borrar_carpeta
)
from cowaver.checkpoints import cargar_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default="checkpoints")
args = parser.parse_args()
data_path = Path(args.data)
checkpoints_path = Path(args.checkpoints)

print("--data", data_path)
print("--checkpoints", checkpoints_path)

tiny_letter_xz_path = Path("tiny-letter-26.tar.xz")
mswc_microset_xz_path = Path("mswc_microset.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
mswc_microset_path = descomprimir_archivo(mswc_microset_xz_path, data_path)
dispositivo = encontrar_dispositivo()
# entrenar_red(
#     net=cowaver,
#     data=TinyMel(tiny_letter_path),
#     num_epochs=3,
#     phase=1,
#     dispositivo=dispositivo,
#     checkpoints_folder=checkpoints_path
# )
def train_cowaver(cowaver: CoWaver):
    entrenar_red(
        net=cowaver,
        data=TinyMel(
            base_dir=tiny_letter_path,
            mel_bins=cowaver.mel_bins,
            position=RandomPosition(mean=0.5, std=0.1)
        ),
        num_epochs=30,
        phase=1,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    entrenar_red(
        net=cowaver,
        data=TinyMel(
            base_dir=tiny_letter_path,
            mel_bins=cowaver.mel_bins,
            transform=Compose([RandomAlign(), RandomScene()]),
            position=RandomPosition(mean=0.5, std=0.1)
        ),
        num_epochs=30,
        phase=2,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    entrenar_red(
        net=cowaver,
        data=TinyMel(
            base_dir=mswc_microset_path,
            mel_bins=cowaver.mel_bins
        ),
        num_epochs=30,
        phase=3,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )

train_cowaver(CoWaver(latent_dim=128, hidden_size=128, mel_bins=40, width_steps=16))
train_cowaver(CoWaver(latent_dim=128, hidden_size=256, mel_bins=40, width_steps=24))
# train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=40, width_steps=24))
# train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=40, width_steps=32))
train_cowaver(CoWaver(latent_dim=256, hidden_size=512, mel_bins=40, width_steps=32))

train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=64, width_steps=24))
# train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=80, width_steps=24))
# train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=80, width_steps=32))
train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=80, width_steps=48))

train_cowaver(CoWaver(latent_dim=384, hidden_size=384, mel_bins=80, width_steps=32))
train_cowaver(CoWaver(latent_dim=512, hidden_size=512, mel_bins=80, width_steps=32))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(mswc_microset_path)
