import argparse
import torch
from pathlib import Path
from cowaver.datamodules import FilteredTestTinyMel, TinyMel
from cowaver.modules import DECODER_REGISTRY, MODEL_REGISTRY, TEMPORAL_ADAPTER_REGISTRY, CoWaver, build_model
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    evaluar_red,
    borrar_carpeta,
    construir_vocabulario_caracteres,
    listar_clases,
)
from cowaver.checkpoints import fases_disponibles, cargar_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default="checkpoints")
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
parser.add_argument("--width-steps", type=int, default=24)
parser.add_argument("--height-bands", type=int, default=4)
parser.add_argument("--seq-len", type=int, default=49)
parser.add_argument("--max-classes", type=int, default=200)
parser.add_argument("--ctc-weight", type=float, default=0.0)
args = parser.parse_args()
if args.max_classes <= 0:
    parser.error("--max-classes must be positive")
if args.ctc_weight < 0:
    parser.error("--ctc-weight must be non-negative")
data_path = Path(args.data)
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
print("--max-classes", args.max_classes)
print("--ctc-weight", args.ctc_weight)

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

def load_cowaver(cowaver: CoWaver):
    letters = TinyMel(
        base_dir=tiny_letter_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1,
        classes=letter_classes,
    )
    phones = TinyMel(
        base_dir=tiny_phones_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1,
        classes=phone_classes,
    )
    words_train = TinyMel(
        base_dir=tiny_mswc_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=2,
        classes=word_train_classes,
    )
    words_test = None
    if len(word_test_classes) > 0:
        words_all = TinyMel(
            base_dir=tiny_mswc_path,
            char_to_idx=char_to_idx,
            mel_bins=cowaver.mel_bins,
            position=RandomPosition(mean=0.5, std=0.1),
            task_id=2,
            classes=word_classes,
        )
        words_test = FilteredTestTinyMel(words_all, word_test_classes)
    for i in range(3):
        cargar_checkpoint(net=cowaver, device=dispositivo, phase=i+1, folder=checkpoints_path)
        eval_cowaver(cowaver, letters, phones, words_train, words_test)

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "width_steps": args.width_steps,
    "height_bands": args.height_bands,
    "seq_len": args.seq_len,
    "decoder": args.decoder,
    "ctc_vocab_size": ctc_vocab_size,
    "ctc_weight": args.ctc_weight,
}
if args.adapter is not None:
    model_kwargs["adapter"] = args.adapter

load_cowaver(build_model(args.architecture, **model_kwargs))
# load_cowaver(CoWaver(latent_dim=384, hidden_size=256, mel_bins=80, width_steps=48))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
