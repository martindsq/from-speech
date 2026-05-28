import argparse
import torch
from pathlib import Path
from torchvision.transforms import Compose
from cowaver.datamodules import FilteredTestTinyMel, TinyMel, MixedTinyMel
from cowaver.modules import DECODER_REGISTRY, MODEL_REGISTRY, TEMPORAL_ADAPTER_REGISTRY, CoWaver, build_model
from cowaver.models import TrainProgramme
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    entrenar_red,
    evaluar_red,
    borrar_carpeta,
    construir_vocabulario_caracteres,
    listar_clases,
)
from cowaver.checkpoints import cargar_checkpoint

parser = argparse.ArgumentParser(
    description=(
        "Train a CoWaver model on the tiny letter, phone, and word datasets "
        "using a three-phase curriculum with configurable data proportions, "
        "CTC auxiliary loss, and linear learning-rate decay."
    )
)
parser.add_argument(
    '--data', '-d',
    default="data",
    help="Directory used to unpack temporary training datasets."
)
parser.add_argument(
    '--checkpoints', '-c',
    default=None,
    help="Directory where phase checkpoints are saved. Omit to skip saving."
)
parser.add_argument(
    "--architecture", "-a",
    choices=sorted(MODEL_REGISTRY),
    default="unconditioned",
    help="Model architecture to train.",
)
parser.add_argument(
    "--decoder", choices=sorted(DECODER_REGISTRY),
    default="convolutional",
    help="Mel decoder architecture.",
)
parser.add_argument(
    "--adapter",
    choices=sorted(TEMPORAL_ADAPTER_REGISTRY),
    default="convolutional",
    help="Temporal adapter architecture between visual features and decoder.",
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
    "--theta-max",
    type=int,
    default=90,
    help="Total number of epochs across all curriculum phases."
)
parser.add_argument(
    "--max-classes",
    type=int,
    default=200,
    help="Maximum number of word classes to include"
)
parser.add_argument(
    "--ctc-weight",
    type=float,
    default=0.1,
    help="Weight of the auxiliary CTC loss. Use 0 to disable it."
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
    default=60,
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
    "--phase1-proportions",
    "-p1",
    nargs=3,
    type=float,
    default=[1.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 1.",
)
parser.add_argument(
    "--phase2-proportions",
    "-p2",
    nargs=3,
    type=float,
    default=[0.20, 0.60, 0.20],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 2.",
)
parser.add_argument(
    "--phase3-proportions",
    "-p3",
    nargs=3,
    type=float,
    default=[0.05, 0.10, 0.85],
    metavar=("LETTERS", "PHONES", "WORDS"),
    help="Sampling proportions for letters, phones, and words in phase 3.",
)
args = parser.parse_args()
if args.theta_max <= 0:
    parser.error("--theta-max must be positive")
if args.theta_max % 3 != 0:
    parser.error("--theta-max must be divisible by 3")
if args.max_classes <= 0:
    parser.error("--max-classes must be positive")
if args.ctc_weight < 0:
    parser.error("--ctc-weight must be non-negative")
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
print("--architecture", args.architecture)
print("--adapter", args.adapter)
print("--decoder", args.decoder)
print("--latent-dim", args.latent_dim)
print("--hidden-size", args.hidden_size)
print("--theta-max", args.theta_max)
print("--max-classes", args.max_classes)
print("--ctc-weight", args.ctc_weight)
print("--epsilon-zero", args.epsilon_zero)
print("--theta", args.theta)
print("--epsilon-theta", args.epsilon_theta)
print("--phase1-proportions", args.phase1_proportions)
print("--phase2-proportions", args.phase2_proportions)
print("--phase3-proportions", args.phase3_proportions)

programme = TrainProgramme(
    theta_max=args.theta_max,
    num_phases=3,
    epsilon_zero=args.epsilon_zero,
    theta=args.theta,
    epsilon_theta=args.epsilon_theta,
)

tiny_letter_xz_path = Path("tiny-letter-30.tar.xz")
tiny_phones_xz_path = Path("tiny-phones-500.tar.xz")
tiny_mswc_xz_path = Path("tiny-mswc-500.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
tiny_phones_path = descomprimir_archivo(tiny_phones_xz_path, data_path)
tiny_mswc_path = descomprimir_archivo(tiny_mswc_xz_path, data_path)
dispositivo = encontrar_dispositivo()

def seleccionar_clases(base_path: Path, max_classes: int | None = None) -> list[str]:
    classes = listar_clases(base_path / "train")
    if max_classes is not None:
        classes = classes[:max_classes]
    return classes

def separar_clases_test(classes: list[str], fraction: float = 0.1, seed: int = 42):
    if len(classes) < 2:
        return classes, []

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(classes), generator=generator).tolist()
    shuffled = [classes[index] for index in permutation]
    test_size = round(len(classes) * fraction)
    test_size = min(max(test_size, 1), len(classes) - 1)
    test = sorted(shuffled[:test_size])
    train = sorted(shuffled[test_size:])
    return train, test

letter_classes = seleccionar_clases(tiny_letter_path)
phone_classes = seleccionar_clases(tiny_phones_path, args.max_classes)
word_classes = seleccionar_clases(tiny_mswc_path, args.max_classes)
word_train_classes, word_test_classes = separar_clases_test(word_classes)

print("--letter-classes", len(letter_classes))
print("--phone-classes", len(phone_classes))
print("--word-train-classes", len(word_train_classes))
print("--word-test-classes", len(word_test_classes))
print("--word-test", word_test_classes)

char_to_idx = construir_vocabulario_caracteres([
    (tiny_letter_path, letter_classes),
    (tiny_phones_path, phone_classes),
    (tiny_mswc_path, word_classes),
])
ctc_vocab_size = len(char_to_idx) + 1
print("--ctc-vocab-size", ctc_vocab_size)

def eval_cowaver(cowaver: CoWaver, letters: TinyMel, phones: TinyMel, words_train: TinyMel, words_test: TinyMel | None = None):
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem} train", end="... ")
    print(evaluar_red(cowaver, words_train))
    if words_test is not None:
        print(f"Evaluando en {tiny_mswc_path.stem} test", end="... ")
        print(evaluar_red(cowaver, words_test))

def make_phase_data(letters: TinyMel, phones: TinyMel, words: TinyMel, proportions: list[float]):
    if any(proportion < 0 for proportion in proportions):
        raise ValueError("phase proportions must be non-negative.")
    datamodules = [letters, phones, words]
    names = ["letters", "phones", "words"]
    active = [
        (data, proportion, name)
        for data, proportion, name in zip(datamodules, proportions, names)
        if proportion > 0
    ]
    if len(active) == 0:
        raise ValueError("at least one phase proportion must be positive.")
    if len(active) == 1:
        return active[0][0]
    return MixedTinyMel(
        datamodules=[data for data, _, _ in active],
        proportions=[proportion for _, proportion, _ in active],
        names=[name for _, _, name in active],
    )

def train_cowaver(cowaver: CoWaver): 
    letters = TinyMel(
        base_dir=tiny_letter_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1,
        classes=letter_classes,
        char_to_idx=char_to_idx,
    )
    phones = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1,
        classes=phone_classes,
        char_to_idx=char_to_idx,
    )
    words_train = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=2,
        classes=word_train_classes,
        char_to_idx=char_to_idx,
    )
    words_test = None
    if len(word_test_classes) > 0:
        words_all = TinyMel(
            base_dir=tiny_mswc_path,
            mel_bins=cowaver.mel_bins,
            position=RandomPosition(mean=0.5, std=0.1),
            task_id=2,
            classes=word_classes,
            char_to_idx=char_to_idx,
        )
        words_test = FilteredTestTinyMel(words_all, word_test_classes)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_train, args.phase1_proportions),
        programme=programme,
        phase=1,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_train, words_test)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_train, args.phase2_proportions),
        programme=programme,
        phase=2,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_train, words_test)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_train, args.phase3_proportions),
        programme=programme,
        phase=3,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_train, words_test)

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": 80,
    "seq_len": 49,
    "decoder": args.decoder,
    "adapter": args.adapter,
    "ctc_vocab_size": ctc_vocab_size,
    "ctc_weight": args.ctc_weight,
}
train_cowaver(build_model(args.architecture, **model_kwargs))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
