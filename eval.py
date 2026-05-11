import argparse
from pathlib import Path
from cowaver.datamodules import TinyMel
from cowaver.modules import CoWaver
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

tiny_letter_xz_path = Path("tiny-letter-26.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
dispositivo = encontrar_dispositivo()
cowaver = CoWaver()
tiny_mel = TinyMel(tiny_letter_path)
for fase in fases_disponibles(cowaver, checkpoints_path):
    cargar_checkpoint(cowaver, dispositivo, fase, checkpoints_path)
    results = evaluar_red(cowaver, tiny_mel)
    print(results)

borrar_carpeta(tiny_letter_path)
