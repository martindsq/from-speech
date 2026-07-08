import argparse
from pathlib import Path
from cowaver.datamodules import FilteredTinyMel, TinyMel
from cowaver.modules import ADAPTER_REGISTRY, ARCHITECTURE_REGISTRY, DECODER_REGISTRY, CoWaver, build_model
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    evaluar_red,
    borrar_carpeta,
    listar_clases,
    seleccionar_clases,
    separar_clases,
)
from cowaver.checkpoints import fases_disponibles, cargar_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default="checkpoints")
parser.add_argument(
    "--architecture",
    "-a",
    choices=sorted(ARCHITECTURE_REGISTRY),
    default="unconditioned",
)
parser.add_argument(
    "--adapter",
    choices=sorted(ADAPTER_REGISTRY),
    default="attn",
)
parser.add_argument(
    "--decoder",
    choices=sorted(DECODER_REGISTRY),
    default="attn",
)
parser.add_argument("--latent-dim", type=int, default=512)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--mel-bins", type=int, default=100)
parser.add_argument("--max-classes", type=int, default=50)
parser.add_argument(
    "--cleanup-data",
    action="store_true",
    help="Delete decompressed dataset folders after evaluation.",
)
args = parser.parse_args()
if args.max_classes < 2:
    parser.error("--max-classes must be at least 2")
if args.mel_bins <= 0:
    parser.error("--mel-bins must be positive")
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
print("--max-classes", args.max_classes)
print("--cleanup-data", args.cleanup_data)

tiny_letter_xz_path = Path("tiny-letter-30.tar.xz")
tiny_phones_xz_path = Path("tiny-phones-500.tar.xz")
tiny_mswc_xz_path = Path("tiny-mswc-500.tar.xz")

tiny_letter_path = descomprimir_archivo(tiny_letter_xz_path, data_path)
tiny_phones_path = descomprimir_archivo(tiny_phones_xz_path, data_path)
tiny_mswc_path = descomprimir_archivo(tiny_mswc_xz_path, data_path)
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

def eval_cowaver(
    cowaver: CoWaver,
    letters: TinyMel,
    phones_train: TinyMel,
    words_train: TinyMel,
    phones_test: TinyMel,
    words_test: TinyMel,
):
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem} train", end="... ")
    print(evaluar_red(cowaver, phones_train))
    print(f"Evaluando en {tiny_phones_path.stem} test", end="... ")
    print(evaluar_red(cowaver, phones_test))
    print(f"Evaluando en {tiny_mswc_path.stem} train", end="... ")
    print(evaluar_red(cowaver, words_train))
    print(f"Evaluando en {tiny_mswc_path.stem} test", end="... ")
    print(evaluar_red(cowaver, words_test))

def load_cowaver(cowaver: CoWaver):
    letters = TinyMel(
        base_dir=tiny_letter_path,
        mel_bins=cowaver.mel_bins,
        task_id=1,
        classes=letter_classes,
    )
    phones_train = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        task_id=1,
        classes=phone_train_classes,
    )
    phones_all = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        task_id=1,
        classes=phone_classes,
    )
    phones_test = FilteredTinyMel(phones_all, phone_test_classes)
    words_train = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        task_id=2,
        classes=word_train_classes,
    )
    words_all = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        task_id=2,
        classes=word_classes,
    )
    words_test = FilteredTinyMel(words_all, word_test_classes)
    phases = fases_disponibles(cowaver, checkpoints_path)
    if not phases:
        raise FileNotFoundError(f"No checkpoints found for '{cowaver.name}' in {checkpoints_path}")
    for phase in sorted(phases):
        cargar_checkpoint(net=cowaver, device=dispositivo, phase=phase, folder=checkpoints_path)
        eval_cowaver(cowaver, letters, phones_train, words_train, phones_test, words_test)

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "seq_len": 49,
    "adapter": args.adapter,
    "decoder": args.decoder,
}
load_cowaver(build_model(args.architecture, **model_kwargs))

if args.cleanup_data:
    borrar_carpeta(tiny_letter_path)
    borrar_carpeta(tiny_phones_path)
    borrar_carpeta(tiny_mswc_path)
