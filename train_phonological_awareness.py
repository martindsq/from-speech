import argparse
from pathlib import Path
from time import perf_counter

from cowaver.datamodules import TinyMel, TinyPairedMel
from cowaver.checkpoints import cargar_checkpoint
from cowaver.modules import ADAPTER_REGISTRY, ARCHITECTURE_REGISTRY, DECODER_REGISTRY, CoWaver, build_model, PhonologicalAwareness
from cowaver.models import TrainProgramme
from cowaver.transforms import RandomPosition
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red,
    borrar_carpeta,
    listar_clases,
    seleccionar_clases,
    separar_clases,
)

parser = argparse.ArgumentParser(
    description=(
        "Train a CoWaver model on the tiny letter, phone, and word datasets "
        "using a three-phase curriculum with configurable data proportions "
        "and linear learning-rate decay."
    )
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
    "--adapter",
    choices=sorted(ADAPTER_REGISTRY),
    default="pointwise",
    help="Temporal adapter architecture between visual features and decoder.",
)
parser.add_argument(
    "--decoder", choices=sorted(DECODER_REGISTRY),
    default="recurrent",
    help="Mel decoder architecture.",
)
parser.add_argument(
    "--latent-dim",
    type=int,
    default=256,
    help="Latent feature dimension used by adapters and decoders."
)
parser.add_argument(
    "--hidden-size",
    type=int,
    default=256,
    help="Hidden size used by decoder modules."
)
parser.add_argument(
    "--mel-bins",
    type=int,
    default=100,
    help="Number of Mel bands used by the training target.",
)
parser.add_argument(
    "--theta-max",
    type=int,
    default=20,
    help="Total number of epochs across all curriculum phases."
)
parser.add_argument(
    "--max-classes",
    type=int,
    default=50,
    help="Maximum number of word classes to include"
)
parser.add_argument(
    "--epsilon-zero",
    "-e0",
    type=float,
    default=3e-4,
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
    default=3e-5,
    help="Learning rate at epoch theta and for the flat tail of training.",
)
args = parser.parse_args()
if args.theta_max <= 0:
    parser.error("--theta-max must be positive")
if args.max_classes < 2:
    parser.error("--max-classes must be at least 2")
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

data_path = Path(args.data)
checkpoints_path = None
if args.checkpoints is not None:
    checkpoints_path = Path(args.checkpoints)
data_path.mkdir(parents=True, exist_ok=True)

print("--data", data_path)
print("--checkpoints", checkpoints_path)
print("--adapter", args.adapter)
print("--decoder", args.decoder)
print("--latent-dim", args.latent_dim)
print("--hidden-size", args.hidden_size)
print("--mel-bins", args.mel_bins)
print("--theta-max", args.theta_max)
print("--max-classes", args.max_classes)
print("--epsilon-zero", args.epsilon_zero)
print("--theta", args.theta)
print("--epsilon-theta", args.epsilon_theta)

programme = TrainProgramme(
    theta_max=args.theta_max,
    num_phases=1,
    epsilon_zero=args.epsilon_zero,
    theta=args.theta,
    epsilon_theta=args.epsilon_theta,
)

tiny_letter_xz_path = Path("tiny-letter-30.tar.xz")
tiny_phones_xz_path = Path("tiny-phones-500.tar.xz")
tiny_mswc_xz_path = Path("tiny-mswc-500.tar.xz")

decompression_started_at = perf_counter()
tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
tiny_phones_path = descomprimir_archivo(tiny_phones_xz_path, data_path)
tiny_mswc_path = descomprimir_archivo(tiny_mswc_xz_path, data_path)
decompression_seconds = perf_counter() - decompression_started_at
print(f"Cronómetro descomprension: {decompression_seconds:.2f}")

dispositivo = encontrar_dispositivo()

letter_classes = listar_clases(tiny_letter_path / "train")
phone_classes = seleccionar_clases(tiny_phones_path, args.max_classes, seed=42)
word_classes = seleccionar_clases(tiny_mswc_path, args.max_classes, seed=42)
phone_train_classes, phone_test_classes = separar_clases(phone_classes)
word_train_classes, word_test_classes = separar_clases(word_classes)

print("--letter-classes", len(letter_classes))
print("--phone-train-classes", len(phone_train_classes))
print("--phone-test-classes", len(phone_test_classes))
print("--phone-test", phone_test_classes)
print("--word-train-classes", len(word_train_classes))
print("--word-test-classes", len(word_test_classes))
print("--word-test", word_test_classes)

# Cargamos el dataset
data = TinyPairedMel(
    phonetized_dir=tiny_phones_path,
    spoken_dir=tiny_mswc_path,
    mel_bins=args.mel_bins,
    classes=phone_train_classes
)

# Cargamos el checkpoint de cowaver
model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "seq_len": 49,
    "decoder": args.decoder,
    "adapter": args.adapter,
}
cowaver = build_model("unconditioned", **model_kwargs)
cargar_checkpoint(
    net=cowaver,
    device=dispositivo,
    phase=1,
    folder=checkpoints_path,
)

# Creamos el modelo de phonological awareness
phonological_awareness = PhonologicalAwareness(cowaver=cowaver, seq_len=49)

"""Entrena una fase e informa su duración total en segundos."""
phase_started_at = perf_counter()
entrenar_red(
    net=phonological_awareness,
    data=data,
    programme=programme,
    phase=1,
    dispositivo=dispositivo,
    checkpoints_folder=checkpoints_path,
)
phase_seconds = perf_counter() - phase_started_at
print(f"Cronómetro: {phase_seconds:.2f}")

#borrar_carpeta(tiny_letter_path)
#borrar_carpeta(tiny_phones_path)
#borrar_carpeta(tiny_mswc_path)
