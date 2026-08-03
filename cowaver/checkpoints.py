import torch
from pathlib import Path
from dataclasses import asdict
from .models import TrainHistory, TrainProgramme, TrainableMixin

def ruta_al_checkpoint(net: TrainableMixin, folder: Path):
    return folder / f"{net.name}.pt"

def guardar_checkpoint(
    net: TrainableMixin,
    train_history: TrainHistory,
    programme: TrainProgramme,
    folder: Path
):
    path = ruta_al_checkpoint(net, folder)
    print(f"Guardando checkpoint en {path}", end="... ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": net.state_dict(),
        "epoch": train_history.num_epochs,
        "optimizer": net.optimizer(programme).state_dict(),
        "extra": {
            "history": asdict(train_history),
            "ckpt": str(path)
        }
    }
    torch.save(ckpt, path)
    print("OK")

def imprimir_encabezado(net: TrainableMixin):
    """Imprime el encabezado del entrenamiento

    Arguments
    ---------
    net : TrainableMixin
    	Red neuronal artifical a entrenar.
    """
    line = "=" * 49
    print(line)
    print(f"Red {net.name}")
    print(line)

def cargar_checkpoint(net: TrainableMixin, folder: Path, silent: bool = False):
    """Carga los pesos de un checkpoint en una red neuronal artificial.

    Arguments
    ---------
    net : TrainableMixin
    	Red neuronal artificial
    folder : Path
    	Ruta a la carpeta donde se encuentran los checkpoints.
    silent : Bool
    	De ser verdadero, no imprime nada. De lo contrario, imprime la historia
    	del entrenamiento cargado.
    """
    path = ruta_al_checkpoint(net, folder)
    dispositivo = torch.device("cpu")
    ckpt = torch.load(path, map_location=dispositivo, weights_only=True)
    net = net.to(dispositivo)
    net.load_state_dict(ckpt["model"], strict=False)
    train_history = TrainHistory(**ckpt["extra"]["history"])
    num_epochs = train_history.num_epochs
    if not silent:
        imprimir_encabezado(net)
    for epoch in range(num_epochs):
        train_loss = train_history.train_losses[epoch]
        val_loss = train_history.val_losses[epoch]
        if not silent:
            print(f"Época {epoch+1}/{num_epochs}", end=" | ")
            print(f"train_loss={train_loss:.4f}", end=" | ")
            print(f"val_loss={val_loss:.4f}")
    return train_history
