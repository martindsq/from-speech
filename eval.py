import argparse
from pathlib import Path
from cowaver.datamodules import TinyMel
from cowaver.modules import DECODER_REGISTRY, MODEL_REGISTRY, TEMPORAL_ADAPTER_REGISTRY, CoWaver, build_model
from cowaver.transforms import RandomAlign, RandomPosition, RandomScene
from cowaver.utils import (
    descomprimir_archivo,
    encontrar_dispositivo,
    evaluar_red,
    borrar_carpeta,
    construir_vocabulario_caracteres,
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

char_to_idx = construir_vocabulario_caracteres([
    (tiny_letter_path, None),
    (tiny_phones_path, args.max_classes),
    (tiny_mswc_path, args.max_classes),
])
ctc_vocab_size = len(char_to_idx) + 1
print("--ctc-vocab-size", ctc_vocab_size)

def eval_cowaver(cowaver: CoWaver, letters: TinyMel, phones: TinyMel, words: TinyMel): 
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem}", end="... ")
    print(evaluar_red(cowaver, words))

def load_cowaver(cowaver: CoWaver):
    letters = TinyMel(
        base_dir=tiny_letter_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1
    )
    phones = TinyMel(
        base_dir=tiny_phones_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1,
        max_classes=args.max_classes,
    )
    words = TinyMel(
        base_dir=tiny_mswc_path,
        char_to_idx=char_to_idx,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=2,
        max_classes=args.max_classes,
    )
    for i in range(3):
        cargar_checkpoint(net=cowaver, device=dispositivo, phase=i+1, folder=checkpoints_path)
        eval_cowaver(cowaver, letters, phones, words)

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
