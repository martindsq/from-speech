from pathlib import Path
from cowaver.datasets import ImageMelDataset
from cowaver.datamodules import TinyMel
from cowaver.modules import CoWaver
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red
)

tiny_letter_filename = Path("data/tiny-letter-26.tar.xz")
tiny_kalulu_filename = Path("data/tiny-kalulu-200.tar.xz")
data_path = Path("data")

tiny_letter_path = descomprimir_archivo(tiny_letter_filename, data_path)
tiny_kalulu_path = descomprimir_archivo(tiny_kalulu_filename, data_path)

letters_data = TinyMel(tiny_letter_path)
words_data = TinyMel(tiny_kalulu_path)

dispositivo = encontrar_dispositivo()
cowaver = CoWaver()
entrenar_red(
    net=cowaver,
    data=letters_data,
    num_epochs=20,
    phase=1,
    dispositivo=dispositivo
)
entrenar_red(
    net=cowaver,
    data=words_data,
    num_epochs=30,
    phase=2,
    dispositivo=dispositivo
)
