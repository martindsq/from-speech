import argparse
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
    borrar_carpeta
)
from cowaver.checkpoints import cargar_checkpoint

parser = argparse.ArgumentParser()
# Available architectures: unconditioned, conditioned, dual-route
# Available temporal adapters: convolutional, recurrent, transformer
# Available decoders: convolutional, recurrent, transformer
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
parser.add_argument(
    "--phase1-proportions",
    nargs=3,
    type=float,
    default=[1.0, 0.0, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
parser.add_argument(
    "--phase2-proportions",
    nargs=3,
    type=float,
    default=[0.25, 0.75, 0.0],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
parser.add_argument(
    "--phase3-proportions",
    nargs=3,
    type=float,
    default=[0.10, 0.15, 0.75],
    metavar=("LETTERS", "PHONES", "WORDS"),
)
args = parser.parse_args()
data_path = Path(args.data)
checkpoints_path = Path(args.checkpoints)
data_path.mkdir(parents=True, exist_ok=True)

print("--data", data_path)
print("--checkpoints", checkpoints_path)
print("--architecture", args.architecture)
print("--adapter", args.adapter)
print("--decoder", args.decoder)
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

def eval_cowaver(cowaver: CoWaver, letters: TinyMel, phones: TinyMel, words: TinyMel):
    print(f"Evaluando en {tiny_letter_path.stem}", end="... ")
    print(evaluar_red(cowaver, letters))
    print(f"Evaluando en {tiny_phones_path.stem}", end="... ")
    print(evaluar_red(cowaver, phones))
    print(f"Evaluando en {tiny_mswc_path.stem}", end="... ")
    print(evaluar_red(cowaver, words))

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
        task_id=1
    )
    phones = TinyMel(
        base_dir=tiny_phones_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=1
    )
    words = TinyMel(
        base_dir=tiny_mswc_path,
        mel_bins=cowaver.mel_bins,
        position=RandomPosition(mean=0.5, std=0.1),
        task_id=2
    )
    entrenar_red(
	net=cowaver,
        data=make_phase_data(letters, phones, words, args.phase1_proportions),
        num_epochs=30,
        phase=1,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words, args.phase2_proportions),
        num_epochs=30,
        phase=2,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)
    entrenar_red(
        net=cowaver,
        data=make_phase_data(letters, phones, words, args.phase3_proportions),
        num_epochs=30,
        phase=3,
        dispositivo=dispositivo,
        checkpoints_folder=checkpoints_path
    )
    eval_cowaver(cowaver, letters, phones, words)

model_kwargs = {
    "latent_dim": 256,
    "hidden_size": 256,
    "mel_bins": 80,
    "width_steps": 24,
    "decoder": args.decoder,
}
if args.adapter is not None:
    model_kwargs["adapter"] = args.adapter

train_cowaver(build_model(args.architecture, **model_kwargs))

borrar_carpeta(tiny_letter_path)
borrar_carpeta(tiny_phones_path)
borrar_carpeta(tiny_mswc_path)
