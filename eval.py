import argparse
from pathlib import Path
from cowaver.datamodules import TinyPairedMel
from cowaver.modules.reading import (
    SpeechAutoEncoder,
    PhonologicalAwareness,
    PhonologicalRoute
)
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    evaluar_red,
    borrar_carpeta,
    listar_clases,
    seleccionar_clases,
    separar_clases,
)
from cowaver.checkpoints import cargar_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--language", "-l",
    choices=["spanish", "frenche"],
    default="spanish",
    help="Language to be trained.",
)
parser.add_argument(
    "--dataset-size",
    type=int,
    default=480,
    help="Size of the dataset to train.",
)
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default="checkpoints")
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--filters", type=int, default=16)
parser.add_argument("--mel-bins", type=int, default=80)

args = parser.parse_args()
LANGUAGE = args.language
DATASET_SIZE = args.dataset_size
MEL_BINS = args.mel_bins
H_DIM = args.hidden_dim
N_FILTERS = args.filters
SEED = 42
CORNET_CKPT_URL = "https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth"

print("LANGUAGE", LANGUAGE)
print("DATASET_SIZE", DATASET_SIZE)
print("MEL_BINS", MEL_BINS)
print("H_DIM", H_DIM)
print("N_FILTERS", N_FILTERS)
print("SEED", SEED)
print("CORNET_CKPT_URL", CORNET_CKPT_URL)

workspace_path = Path(LANGUAGE)
compressed_phones_path = workspace_path / f"kalulu-phones-{DATASET_SIZE}.tar.xz"
compressed_spoken_path = workspace_path / f"kalulu-spoken-{DATASET_SIZE}.tar.xz"

print(f"Buscando {compressed_phones_path}", end="... ")
print("OK" if compressed_phones_path.exists() else "ERROR")

print(f"Buscando {compressed_spoken_path}", end="... ")
print("OK" if compressed_spoken_path.exists() else "ERROR")

data_path = Path(args.data)
checkpoints_path = None
if args.checkpoints is not None:
    checkpoints_path = Path(args.checkpoints)
data_path.mkdir(parents=True, exist_ok=True)

print("--data", data_path)
print("--checkpoints", checkpoints_path)

phones_path = descomprimir_archivo(compressed_phones_path, data_path)
spoken_path = descomprimir_archivo(compressed_spoken_path, data_path)

palabras = listar_clases(phones_path / "train")
print("Total de palabras:", len(palabras))

palabras_a_entrenar, palabras_a_generalizar = separar_clases(palabras, fraction=0.2)
print("Palabras a entrenar:", len(palabras_a_entrenar))
print(f"Palabras a generalizar ({len(palabras_a_generalizar)}):", palabras_a_generalizar)

full_data = TinyPairedMel(
    phones_path,
    spoken_path,
    mel_bins=MEL_BINS,
    classes=palabras
)
training_data = TinyPairedMel(
    phones_path,
    spoken_path,
    mel_bins=MEL_BINS,
    classes=palabras_a_entrenar
)
generalization_data = TinyPairedMel(
    phones_path,
    spoken_path,
    mel_bins=MEL_BINS,
    classes=palabras_a_generalizar
)

print("Evaluando Speech AutoEncoder...")
speech_autoencoder = SpeechAutoEncoder(h_dim=H_DIM, n_filters=N_FILTERS)
cargar_checkpoint(speech_autoencoder, folder=checkpoints_path, silent=True)
print(evaluar_red(speech_autoencoder, full_data))

print("Evaluando Phonological Awareness...")
phonological_awareness = PhonologicalAwareness(h_dim=H_DIM, n_filters=N_FILTERS)
cargar_checkpoint(phonological_awareness, folder=checkpoints_path, silent=True)
print(evaluar_red(phonological_awareness, training_data))
print(evaluar_red(phonological_awareness, generalization_data))

print("Evaluando Phonological Route...")
phonological_route = PhonologicalRoute(h_dim=H_DIM, n_filters=N_FILTERS)
cargar_checkpoint(phonological_route, folder=checkpoints_path, silent=True)
print(evaluar_red(phonological_route, training_data))
print(evaluar_red(phonological_route, generalization_data))

borrar_carpeta(spoken_path)
borrar_carpeta(phones_path)
