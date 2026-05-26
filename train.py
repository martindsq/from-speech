import argparse
import torch
from pathlib import Path
from torchvision.transforms import Compose
from cowaver.datamodules import TinyMel, MixedTinyMel
from cowaver.modules import DECODER_REGISTRY, MODEL_REGISTRY, TEMPORAL_ADAPTER_REGISTRY, CoWaver, build_model
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

parser = argparse.ArgumentParser()
# Available architectures: unconditioned, conditioned, dual-route
# Available temporal adapters: convolutional, recurrent, transformer
# Available decoders: convolutional, recurrent, transformer
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default=None)
parser.add_argument(
    "--architecture",
    "-a",
    choices=sorted(MODEL_REGISTRY),
    default="unconditioned",
)
parser.add_argument(
    "--decoder",
    choices=sorted(DECODER_REGISTRY),
    default="convolutional",
)
parser.add_argument(
    "--adapter",
    choices=sorted(TEMPORAL_ADAPTER_REGISTRY),
    default="convolutional",
)
parser.add_argument("--latent-dim", type=int, default=256)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--mel-bins", type=int, default=80)
parser.add_argument("--width-steps", "-ws", type=int, default=24)
parser.add_argument("--height-bands", "-hb", type=int, default=4)
parser.add_argument("--seq-len", type=int, default=49)
parser.add_argument("--max-epochs", type=int, default=30)
parser.add_argument("--max-classes", type=int, default=200)
parser.add_argument("--ctc-weight", type=float, default=0.0)
parser.add_argument(
    "--phase1-proportions",
    "-p1",
    nargs=3,
    type=float,
    default=[1.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
parser.add_argument(
    "--phase2-proportions",
    "-p2",
    nargs=3,
    type=float,
    default=[0.25, 0.75, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
parser.add_argument(
    "--phase3-proportions",
    "-p3",
    nargs=3,
    type=float,
    default=[0.10, 0.15, 0.75],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
args = parser.parse_args()
if args.max_epochs <= 0:
    parser.error("--max-epochs must be positive")
if args.max_classes <= 0:
    parser.error("--max-classes must be positive")
if args.ctc_weight < 0:
    parser.error("--ctc-weight must be non-negative")

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
print("--width-steps", args.width_steps)
print("--height-bands", args.height_bands)
print("--seq-len", args.seq_len)
print("--max-epochs", args.max_epochs)
print("--max-classes", args.max_classes)
print("--ctc-weight", args.ctc_weight)
print("--phase1-proportions", args.phase1_proportions)
print("--phase2-proportions", args.phase2_proportions)
print("--phase3-proportions", args.phase3_proportions)

tiny_letter_xz_path = Path("tiny-letter-30.tar.xz")
tiny_phones_xz_path = Path("tiny-phones-200.tar.xz")
tiny_mswc_xz_path = Path("tiny-mswc-200.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
tiny_phones_path = descomprimir_archivo(tiny_phones_xz_path, data_path)
tiny_mswc_path = descomprimir_archivo(tiny_mswc_xz_path, data_path)
dispositivo = encontrar_dispositivo()

def seleccionar_clases(base_path: Path, max_classes: int | None = None) -> list[str]:
    classes = listar_clases(base_path / "train")
    if max_classes is not None:
        classes = classes[:max_classes]
    return classes

def separar_clases_heldout(classes: list[str], fraction: float = 0.1, seed: int = 42):
    if len(classes) < 2:
        return classes, []

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(classes), generator=generator).tolist()
    shuffled = [classes[index] for index in permutation]
    heldout_size = round(len(classes) * fraction)
    heldout_size = min(max(heldout_size, 1), len(classes) - 1)
    heldout = sorted(shuffled[:heldout_size])
    seen = sorted(shuffled[heldout_size:])
    return seen, heldout

letter_classes = seleccionar_clases(tiny_letter_path)
phone_classes = seleccionar_clases(tiny_phones_path, args.max_classes)
word_classes = seleccionar_clases(tiny_mswc_path, args.max_classes)
word_seen_classes, word_heldout_classes = separar_clases_heldout(word_classes)

print("--letter-classes", len(letter_classes))
print("--phone-classes", len(phone_classes))
print("--word-seen-classes", len(word_seen_classes))
print("--word-heldout-classes", len(word_heldout_classes))
print("--word-heldout", word_heldout_classes)

char_to_idx = construir_vocabulario_caracteres([
    (tiny_letter_path, letter_classes),
    (tiny_phones_path, phone_classes),
    (tiny_mswc_path, word_classes),
])
ctc_vocab_size = len(char_to_idx) + 1
print("--ctc-vocab-size", ctc_vocab_size)

def eval_cowaver(cowaver: CoWaver, letters: TinyMel, phones: TinyMel, words_seen: TinyMel, words_heldout: TinyMel | None = None):
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem} seen", end="... ")
    print(evaluar_red(cowaver, words_seen))
    if words_heldout is not None:
        print(f"Evaluando en {tiny_mswc_path.stem} heldout", end="... ")
        print(evaluar_red(cowaver, words_heldout))

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
    words_seen = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=2,
        classes=word_seen_classes,
        char_to_idx=char_to_idx,
    )
    words_heldout = None
    if len(word_heldout_classes) > 0:
        words_heldout = TinyMel(
            base_dir=tiny_mswc_path,
            mel_bins=cowaver.mel_bins,
            position=RandomPosition(mean=0.5, std=0.1),
            task_id=2,
            classes=word_heldout_classes,
            char_to_idx=char_to_idx,
        )
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_seen, args.phase1_proportions),
        num_epochs=args.max_epochs,
        phase=1,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_seen, words_heldout)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_seen, args.phase2_proportions),
        num_epochs=args.max_epochs,
        phase=2,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_seen, words_heldout)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words_seen, args.phase3_proportions),
        num_epochs=args.max_epochs,
        phase=3,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words_seen, words_heldout)

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "width_steps": args.width_steps,
    "height_bands": args.height_bands,
    "seq_len": args.seq_len,
    "decoder": args.decoder,
    "adapter": args.adapter,
    "ctc_vocab_size": ctc_vocab_size,
    "ctc_weight": args.ctc_weight,
}

train_cowaver(build_model(args.architecture, **model_kwargs))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
