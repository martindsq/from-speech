from pathlib import Path
from torchvision.transforms import Compose
from cowaver.datamodules import TinyMel
from cowaver.modules import CoWaver
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red
)

tiny_letter_filename = Path("tiny-letter-26.tar.xz")
data_path = Path("data")

tiny_letter_path = descomprimir_archivo(tiny_letter_filename, data_path)

dispositivo = encontrar_dispositivo()
cowaver = CoWaver()
entrenar_red(
    net=cowaver,
    data=TinyMel(tiny_letter_path),
    num_epochs=30,
    phase=1,
    dispositivo=dispositivo
)
entrenar_red(
    net=cowaver,
    data=TinyMel(
        tiny_letter_path,
        transform=RandomAlign((0.2, 0.8)),
        position=RandomPosition(mean=0.5, std=0.1),
    ),
    num_epochs=30,
    phase=2,
    dispositivo=dispositivo
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
    num_epochs=30,
    phase=3,
    dispositivo=dispositivo
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
    num_epochs=30,
    phase=4,
    dispositivo=dispositivo
)
