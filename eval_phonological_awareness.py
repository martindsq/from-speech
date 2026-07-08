import argparse
from pathlib import Path
from cowaver.datamodules import TinyPairedMel
from cowaver.modules import (
    ADAPTER_REGISTRY,
    DECODER_REGISTRY,
    PhonologicalAwareness,
    CoWaver,
    build_model
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
from cowaver.checkpoints import fases_disponibles, cargar_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default="data")
parser.add_argument('--checkpoints', '-c', default="checkpoints")
parser.add_argument(
    "--adapter",
    choices=sorted(ADAPTER_REGISTRY),
    default="pointwise",
)
parser.add_argument(
    "--decoder",
    choices=sorted(DECODER_REGISTRY),
    default="recurrent",
)
parser.add_argument("--latent-dim", type=int, default=256)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--mel-bins", type=int, default=100)
parser.add_argument("--max-classes", type=int, default=50)
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
print("--adapter", args.adapter)
print("--decoder", args.decoder)
print("--latent-dim", args.latent_dim)
print("--hidden-size", args.hidden_size)
print("--mel-bins", args.mel_bins)
print("--max-classes", args.max_classes)

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

model_kwargs = {
    "latent_dim": args.latent_dim,
    "hidden_size": args.hidden_size,
    "mel_bins": args.mel_bins,
    "seq_len": 49,
    "adapter": args.adapter,
    "decoder": args.decoder,
}
cowaver = build_model("unconditioned", **model_kwargs)
phonological_awareness = PhonologicalAwareness(cowaver=cowaver, seq_len=49)
cargar_checkpoint(
    net=phonological_awareness,
    device=dispositivo,
    phase=1,
    folder=checkpoints_path
)

words_train = TinyPairedMel(
    phonetized_dir=tiny_phones_path,
    spoken_dir=tiny_mswc_path,
    mel_bins=args.mel_bins,
    classes=word_train_classes
)
words_test = TinyPairedMel(
    phonetized_dir=tiny_phones_path,
    spoken_dir=tiny_mswc_path,
    mel_bins=args.mel_bins,
    classes=word_test_classes
)
print(f"Evaluando en {tiny_mswc_path.stem} train", end="... ")
print(evaluar_red(phonological_awareness, words_train))
print(f"Evaluando en {tiny_mswc_path.stem} test", end="... ")
print(evaluar_red(phonological_awareness, words_test))

# borrar_carpeta(tiny_letter_path)
# borrar_carpeta(tiny_phones_path)
# borrar_carpeta(tiny_mswc_path)
