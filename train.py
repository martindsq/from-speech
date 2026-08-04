import argparse
from pathlib import Path

import cowaver
from cowaver.datamodules import TinyPairedMel
from cowaver.checkpoints import cargar_checkpoint, guardar_checkpoint
from cowaver.models import TrainProgramme
from cowaver.modules.reading import (
    SpeechAutoEncoder,
    PhonologicalAwareness,
    PhonologicalRoute
)
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    sembrar_semilla,
    entrenar_red,
    borrar_carpeta,
    listar_clases,
    separar_clases,
)

parser = argparse.ArgumentParser(description=("Train reading models."))
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
parser.add_argument(
    '--data', '-d',
    default="data",
    help="Directory used to unpack temporary training datasets."
)
parser.add_argument(
    '--checkpoints', '-c',
    default="checkpoints",
    help="Directory where phase checkpoints are saved. Omit to skip saving."
)
parser.add_argument(
    "--hidden-dim",
    type=int,
    default=128,
    help="Size of the autoencoder bottleneck.",
)
parser.add_argument(
    "--filters",
    type=int,
    default=16,
    help="Number of filters in the autoencoder convolutional layers.",
)
parser.add_argument(
    "--mel-bins",
    type=int,
    default=80,
    help="Number of Mel bands used by the training target.",
)
parser.add_argument(
    "--theta-max",
    type=int,
    default=20,
    help="Total number of epochs across all curriculum phases."
)
parser.add_argument(
    "--epsilon-zero",
    "-e0",
    type=float,
    default=1e-3,
    help="Initial learning rate.",
)
parser.add_argument(
    "--theta",
    "-t",
    type=int,
    default=20,
    help="Global epoch where the learning-rate decay reaches epsilon_theta.",
)
parser.add_argument(
    "--epsilon-theta",
    "-et",
    type=float,
    default=1e-4,
    help="Learning rate at epoch theta and for the flat tail of training.",
)

args = parser.parse_args()
if args.theta_max <= 0:
    parser.error("--theta-max must be positive")
if args.mel_bins <= 0:
    parser.error("--mel-bins must be positive")
if args.epsilon_zero <= 0:
    parser.error("--epsilon-zero must be positive")
if args.theta <= 0:
    parser.error("--theta must be positive")
if args.theta > args.theta_max:
    parser.error("--theta must be less than or equal to --theta-max")
if args.epsilon_theta <= 0 or args.epsilon_theta > args.epsilon_zero:
    parser.error("--epsilon-theta must be in the range (0, epsilon-zero]")

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
print("--theta-max", args.theta_max)
print("--epsilon-zero", args.epsilon_zero)
print("--theta", args.theta)
print("--epsilon-theta", args.epsilon_theta)

programme = TrainProgramme(
    theta_max=args.theta_max,
    epsilon_zero=args.epsilon_zero,
    theta=args.theta,
    epsilon_theta=args.epsilon_theta,
    patience=5
)

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

# Speech AutoEncoder

sembrar_semilla(SEED)
speech_autoencoder = SpeechAutoEncoder(h_dim=H_DIM, n_filters=N_FILTERS)
speech_autoencoder, train_history = entrenar_red(
    net=speech_autoencoder,
    data=full_data,
    programme=programme
)
if checkpoints_path is not None:
    guardar_checkpoint(
        speech_autoencoder,
        train_history=train_history,
        programme=programme,
        folder=checkpoints_path
    )

# Phonological Awareness

sembrar_semilla(SEED)
phonological_awareness = PhonologicalAwareness(h_dim=H_DIM, n_filters=N_FILTERS)
phonological_awareness.load_state_dict(speech_autoencoder.state_dict())
phonological_awareness, train_history = entrenar_red(
    net=phonological_awareness,
    data=training_data,
    programme=programme
)
if checkpoints_path is not None:
    guardar_checkpoint(
        phonological_awareness,
        train_history=train_history,
        programme=programme,
        folder=checkpoints_path
    )

# Phonological Route

sembrar_semilla(SEED)
phonological_route = PhonologicalRoute(h_dim=H_DIM, n_filters=N_FILTERS)

checkpoint = torch.utils.model_zoo.load_url(
    CORNET_CKPT_URL,
    map_location=torch.device("cpu")
)
phonological_route.cornet.load_state_dict(checkpoint["state_dict"], strict=False)

checkpoint = phonological_awareness.state_dict()
phonological_route.phonological_awareness.load_state_dict(checkpoint)

print("Midiendo estadística", end="... ")
all_h = []
phonological_route.phonological_awareness.eval()
with torch.no_grad():
    for batch in training_data.train_loader():
        (_, phonetized_mel, _), labels = batch
        phonetized_mel = phonetized_mel.transpose(2, 3)
        h = phonological_route.phonological_awareness.encoder(phonetized_mel)
        all_h.append(h)
all_h = torch.cat(all_h, dim=0)
h_mean = all_h.mean(dim=0)
h_std = all_h.std(dim=0)
phonological_route.set_h_stats(h_mean, h_std)
print("OK")

phonological_route, train_history = entrenar_red(
    net=phonological_route,
    data=training_data,
    programme=programme
)
if checkpoints_path is not None:
    guardar_checkpoint(
        phonological_route,
        train_history=train_history,
        programme=programme,
        folder=checkpoints_path
    )

borrar_carpeta(spoken_path)
borrar_carpeta(phones_path)
