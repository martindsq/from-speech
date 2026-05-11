from torch import load, device, save
from pathlib import Path
from dataclasses import asdict
from .modules import TrainableModule
from .models import TrainHistory

CHECKPOINTS_ROOT = Path('checkpoints')

def fase_mas_alta(net: TrainableModule, folder: Path) -> int:
    prefix = f"{net.name}_"
    suffix = "_phase.pt"
    phases = []

    for path in folder.glob(f"{net.name}_*_phase.pt"):
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue

        phase_token = name[len(prefix):-len(suffix)]
        phase_number = phase_token.rstrip("tsnrhtdd")
        if phase_number.isdigit():
            phases.append(int(phase_number))

    if not phases:
        raise FileNotFoundError(f"No checkpoints found for '{net.name}' in {folder}")

    return max(phases)

def ruta_al_checkpoint(net: TrainableModule, phase: int, folder: Path):
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    return folder / f"{net.name}_{ordinal(phase)}_phase.pt"

def guardar_checkpoint(net: TrainableModule, train_history: TrainHistory, phase: int = 1, folder: Path = CHECKPOINTS_ROOT):
    path = ruta_al_checkpoint(net, phase, folder)
    print(f"Guardando checkpoint en {path}", end="...")
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": net.state_dict(),
        "epoch": train_history.num_epochs,
        "optimizer": net.optimizer(phase).state_dict(),
        "extra": {
            "history": asdict(train_history),
            "ckpt": str(path)
        }
    }
    save(ckpt, path)
    print("OK")

def imprimir_encabezado(net: TrainableModule, phase: int = 1):
    line = "═" * 80
    print(line)
    print(f"Red {net.name} | fase {phase}")
    print(line)

def cargar_checkpoint(net: TrainableModule, device: device = device("cpu"), phase: int | None = None, folder: Path = CHECKPOINTS_ROOT, silent: bool = False):
    if phase is None:
        phase = fase_mas_alta(net)

    path = ruta_al_checkpoint(net, phase, folder)
    ckpt = load(path, map_location=device, weights_only=True)
    net = net.to(device)
    net.load_state_dict(ckpt["model"], strict=False)
    train_history = TrainHistory(**ckpt["extra"]["history"])
    num_epochs = train_history.num_epochs
    if not silent:
        imprimir_encabezado(net, phase)
    for epoch in range(num_epochs):
        train_loss = train_history.train_losses[epoch]
        val_loss = train_history.val_losses[epoch]
        if not silent:
            print(f"Época {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
    return train_history
