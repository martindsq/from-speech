import argparse
from pathlib import Path
from torchvision.transforms import Compose
from cowaver.datamodules import TinyMel, MixedTinyMel
from cowaver.modules import CoWaver
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red,
    evaluar_red,
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

tiny_letter_xz_path = Path("tiny-letter-30.tar.xz")
tiny_phones_xz_path = Path("tiny-phones-200.tar.xz")
tiny_mswc_xz_path = Path("tiny-mswc-200.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
tiny_phones_path = descomprimir_archivo(tiny_phones_xz_path, data_path)
tiny_mswc_path = descomprimir_archivo(tiny_mswc_xz_path, data_path)
dispositivo = encontrar_dispositivo()

def eval_cowaver(cowaver: CoWaver, letters: TinyMel, phones: TinyMel, words: TinyMel):
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem}", end="... ")
    print(evaluar_red(cowaver, words))

def train_cowaver(cowaver: CoWaver): 
    letters = TinyMel(
        base_dir=tiny_letter_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1)
    )
    phones = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1)
    )
    words = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1)
    )
    entrenar_red(
	net=cowaver,
        data=letters,
        num_epochs=30,
        phase=1,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)
    entrenar_red(
        net=cowaver,
        data=MixedTinyMel(
            datamodules=[letters, phones],
            proportions=[0.25, 0.75],
            names=["letters", "phones"]
	),
        num_epochs=30,
        phase=2,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)
    entrenar_red(
        net=cowaver,
        data=MixedTinyMel(
            datamodules=[letters, phones, words],
            proportions=[0.10, 0.15, 0.75],
            names=["letters", "phones", "words"]
	),
        num_epochs=30,
        phase=3,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)

train_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=80, width_steps=24))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
