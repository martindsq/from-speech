import argparse
from pathlib import Path
from time import perf_counter

from cowaver.datamodules import TinyMel, MixedTinyMel
from cowaver.checkpoints import cargar_checkpoint
from cowaver.modules import ADAPTER_REGISTRY, ARCHITECTURE_REGISTRY, DECODER_REGISTRY, CoWaver, build_model
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
    "--architecture", "-a",
    choices=sorted(ARCHITECTURE_REGISTRY),
    default="unconditioned",
    help="Model architecture to train.",
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
parser.add_argument(
    "--phase1",
    "-p1",
    nargs=3,
    type=float,
    default=[1.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 1.",
)
parser.add_argument(
    "--phase2",
    "-p2",
    nargs=3,
    type=float,
    default=[0.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 2.",
)
parser.add_argument(
    "--phase3",
    "-p3",
    nargs=3,
    type=float,
    default=[0.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 3.",
)
parser.add_argument(
    "--phase",
    type=int,
    choices=(1, 2, 3),
    default=1,
    help="Curriculum phase to train. Phases greater than 1 load the previous phase checkpoint first.",
)
args = parser.parse_args()
if any(
    proportion < 0
    for proportions in (args.phase1, args.phase2, args.phase3)
    for proportion in proportions
):
    parser.error("phase proportions must be non-negative")
phase1_enabled = any(proportion > 0 for proportion in args.phase1)
phase2_enabled = any(proportion > 0 for proportion in args.phase2)
phase3_enabled = any(proportion > 0 for proportion in args.phase3)
if not phase1_enabled:
    parser.error("--phase1must be non-zero")
if phase3_enabled and not phase2_enabled:
    parser.error("--phase3 requires --phase2 to be non-zero")
selected_phase_enabled = {
    1: phase1_enabled,
    2: phase2_enabled,
    3: phase3_enabled,
}[args.phase]
if not selected_phase_enabled:
    parser.error(f"--phase{args.phase} must be non-zero to train phase {args.phase}")
num_phases = 1 + int(phase2_enabled) + int(phase3_enabled)
if args.theta_max <= 0:
    parser.error("--theta-max must be positive")
if args.theta_max % num_phases != 0:
    parser.error(f"--theta-max must be divisible by {num_phases}")
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
if args.phase > 1 and args.checkpoints is None:
    parser.error("--phase greater than 1 requires --checkpoints to load the previous phase")

data_path = Path(args.data)
checkpoints_path = None
if args.checkpoints is not None:
    checkpoints_path = Path(args.checkpoints)
data_path.mkdir(parents=True, exist_ok=True)

print("--data", data_path)
print("--checkpoints", checkpoints_path)
print("--architecture", args.architecture)
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
print("--phase1", args.phase1)
print("--phase2", args.phase2)
print("--phase3", args.phase3)
print("--phase", args.phase)
print("--phase1-enabled", phase1_enabled)
print("--phase2-enabled", phase2_enabled)
print("--phase3-enabled", phase3_enabled)
print("--num-phases", num_phases)

programme = TrainProgramme(
    theta_max=args.theta_max,
    num_phases=num_phases,
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

def make_phase_data(letters: TinyMel, phones: TinyMel, words: TinyMel, proportions: list[float]):
    """Construye el datamodule de una fase a partir de sus proporciones activas."""
    datamodules = [letters, phones, words]
    names = ["letters", "phones", "words"]
    active = [
        (data, proportion, name)
        for data, proportion, name in zip(datamodules, proportions, names)
        if proportion > 0
    ]
    if len(active) == 1:
        return active[0][0]
    return MixedTinyMel(
        datamodules=[data for data, _, _ in active],
        proportions=[proportion for _, proportion, _ in active],
        names=[name for _, _, name in active],
    )


def train_phase(cowaver: CoWaver, data, phase: int):
    """Entrena una fase e informa su duración total en segundos."""
    phase_started_at = perf_counter()
    entrenar_red(
        net=cowaver,
        data=data,
        programme=programme,
        phase=phase,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path,
    )
    phase_seconds = perf_counter() - phase_started_at
    print(f"Phase {phase} stopwatch: {phase_seconds:.2f}")

def train_cowaver(cowaver: CoWaver):
    letters = TinyMel(
        base_dir=tiny_letter_path,
        mel_bins=cowaver.mel_bins,
        task_id=1,
        position=RandomPosition(center=0.5, spread=0.5, axis="x"),
        classes=letter_classes,
    )
    phones_train = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        task_id=1,
        classes=phone_train_classes,
    )
    words_train = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        task_id=2,
        classes=word_train_classes,
    )
    phase_proportions = {
        1: args.phase1,
        2: args.phase2,
        3: args.phase3,
    }
    phase_data = make_phase_data(
        letters,
        phones_train,
        words_train,
        phase_proportions[args.phase],
    )
    if args.phase > 1:
        checkpoint_phase = args.phase - 1
        print(f"Cargando checkpoint de fase {checkpoint_phase}")
        cargar_checkpoint(
            net=cowaver,
            device=dispositivo,
            phase=checkpoint_phase,
            folder=checkpoints_path,
        )
    train_phase(cowaver, phase_data, phase=args.phase)

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "seq_len": 49,
    "decoder": args.decoder,
    "adapter": args.adapter,
}
train_cowaver(build_model(args.architecture, **model_kwargs))

#borrar_carpeta(tiny_letter_path)
#borrar_carpeta(tiny_phones_path)
#borrar_carpeta(tiny_mswc_path)
