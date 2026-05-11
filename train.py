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
# entrenar_red(
#     net=cowaver,
#     data=TinyMel(tiny_letter_path),
#     num_epochs=3,
#     phase=1,
#     dispositivo=dispositivo,
#     checkpoints_folder=checkpoints_path
# )
entrenar_red(
    net=cowaver,
    data=TinyMel(tiny_letter_path),
    num_epochs=20,
    phase=1,
    dispositivo=dispositivo,
    checkpoints_folder=checkpoints_path
)
entrenar_red(
    net=cowaver,
    data=TinyMel(
        tiny_letter_path,
        transform=RandomAlign((0.2, 0.8)),
        position=RandomPosition(mean=0.5, std=0.1),
    ),
    num_epochs=20,
    phase=2,
    dispositivo=dispositivo,
    checkpoints_folder=checkpoints_path
)
entrenar_red(
    net=cowaver,
    data=TinyMel(
        tiny_letter_path,
        transform=Compose([
            RandomAlign((0.2, 0.8)),
            RandomScene(),
        ]),
        position=RandomPosition(mean=0.5, std=0.1),
    ),
    num_epochs=20,
    phase=3,
    dispositivo=dispositivo,
    checkpoints_folder=checkpoints_path
)
entrenar_red(
    net=cowaver,
    data=TinyMel(
        tiny_letter_path,
        transform=Compose([
            RandomAlign((0.1, 0.9)),
            RandomScene(),
        ]),
        position=RandomPosition(mean=0.5, std=0.25),
    ),
    num_epochs=20,
    phase=4,
    dispositivo=dispositivo,
    checkpoints_folder=checkpoints_path
)
borrar_carpeta(tiny_letter_path)
