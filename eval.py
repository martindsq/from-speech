import argparse
from pathlib import Path
from cowaver.datamodules import TinyMel
from cowaver.modules import CoWaver
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    evaluar_red,
    borrar_carpeta
)
from cowaver.checkpoints import fases_disponibles, cargar_checkpoint

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

def eval_cowaver(cowaver: CoWaver):
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
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem}", end="... ")
    print(evaluar_red(cowaver, words))

eval_cowaver(CoWaver(latent_dim=256, hidden_size=256, mel_bins=80, width_steps=24))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
