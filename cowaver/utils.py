import torch
import torchaudio
from torch import Tensor, nn, device
import torch.nn.functional as F
from torchaudio.transforms import Resample
from PIL import Image, ImageDraw, ImageFont
import tarfile
import ipywidgets as widgets
from pathlib import Path
from typing import Any, Callable
from torchvision.transforms import ToTensor
from torchaudio.transforms import MelSpectrogram
from .checkpoints import imprimir_encabezado, guardar_checkpoint
from .models import DataModule, TestResults, TrainHistory, TrainableModule

AUDIO_SAMPLE_RATE = 16_000

def encontrar_dispositivo(silent: bool = False):
    """Encuentra un dispositivo apropiado para entrenar o evaluar una red.

    Parameters
    ----------
    silent: bool 
        No imprime nada en consola, de lo contrario, imprime el dispositivo
        encontrado.

    Returns
    -------
    dispositivo: device 
        El dispositivo encontrado.
    """
    if torch.cuda.is_available():
        dispositivo = device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dispositivo = device("mps")
    else:
        dispositivo = device("cpu")
    if not silent:
        print("Dispositivo encontrado:", dispositivo)
    return dispositivo

def mover_a_dispositivo(x: Tensor | tuple, dispositivo: device | None = None):
    """Mueve un tensor a un dispositivo apropiado para entrenar o evaluar.

    Parameters
    ----------
    x: Tensor | tuple
        Un tensor cualquiera, o una tupla cuyos elementos sean tensores.
    dispositivo: device | None
        Dispositivo apropiado. Si es None, entonces se encuentra uno.

    Returns
    -------
    x: Tensor | tuple
        Igual a x pasado como parámetro, pero ubicado en el dispositivo
        apropiado.
    """
    if dispositivo is None:
        dispositivo = encontrar_dispositivo(silent=True)
    if torch.is_tensor(x):
        return x.to(dispositivo)
    if isinstance(x, tuple):
        return tuple(mover_a_dispositivo(v, dispositivo) for v in x)
    return x

def descomprimir_archivo(archivo: Path, carpeta: Path) -> Path:
    ruta = carpeta / Path(archivo.stem).stem
    if not ruta.exists():
        print("Descomprimiendo", f"{ruta}...")
        with tarfile.open(archivo, mode="r:xz") as tar:
            tar.extractall(path=carpeta)
    else:
        print("La carpeta", ruta, "ya existe.")
    return ruta

def cargar_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze()
    if sample_rate != AUDIO_SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=AUDIO_SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform

def explorar_datos(data: DataModule, f: Callable[[DataModule, Tensor], None]):
    loader = data.inference_loader()
    batches = list(loader)
    labels = [data.labels_from_batch(batch).item() for batch in batches]
    options = [(f"{data.classes[label]}", i) for i, label in enumerate(labels)]
    out = widgets.Output()
    def view(batch_idx):
        batch = batches[batch_idx]
        with out:
            out.clear_output(wait=True)
            f(data, batch)
    w = widgets.interactive(
        view,
        batch_idx=widgets.Dropdown(
            options=options,
            value=0 if batches else None,
            description="Ver:",
            disabled=(len(batches) == 0),
        )
    )
    return widgets.VBox([w, out])

def explorar_red(net: TrainableModule, data: DataModule, f: Callable[[TrainableModule, Any], ...]):
    loader = data.inference_loader()
    batches = list(loader)
    labels = [data.labels_from_batch(batch).item() for batch in batches]
    options = [(f"{data.classes[label]}", i) for i, label in enumerate(labels)]
    out = widgets.Output()
    dispositivo = encontrar_dispositivo(silent=True)
    net = mover_a_dispositivo(net, dispositivo)
    def view(batch_idx):
        batch = mover_a_dispositivo(batches[batch_idx], dispositivo)
        with torch.no_grad():
            with out:
                out.clear_output(wait=True)
                f(net, batch)
    w = widgets.interactive(
        view,
        batch_idx=widgets.Dropdown(
            options=options,
            value=0 if batches else None,
            description="Ver:",
            disabled=(len(batches) == 0),
        )
    )
    return widgets.VBox([w, out])

def extract_mel(waveform: Tensor, mel_bins: int = 40):
    """Calcula un espectrograma Mel.

    Parameters
    ----------
    waveform: Tensor
        Tensor de forma [B, T] donde B es el tamaño del batch y T el número de
        frames temporales. Se puede omitir B.

    Returns
    -------
    mel: Tensor
        Espectrograma Mel de forma [B, mel_bins, T]. B es igual a 1 si se
        omitió en waveform.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    n_fft = 2048
    wav2vec_hz = 49
    hop_length = int(round(AUDIO_SAMPLE_RATE / wav2vec_hz))

    # Create MelSpectrogram transform
    mel_transform = MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mel_bins,
        f_min=0.0,
        f_max=AUDIO_SAMPLE_RATE / 2
    ).to(waveform.device)

    # Torchaudio expects [channels, time], so we treat batch as "channels"
    mel = mel_transform(waveform)  # [B, n_mels, time_frames]

    return mel

def clip_waveform(waveform: Tensor, duration: float = 1.0):
    """Corta el audio o rellena con ceros para que la duracion sea la
    especificada.

    Parameters
    ----------
    waveform: Tensor
        Tensor de forma [T] donde T el número de frames temporales.
    duracion: float
        Duración, en segundos, del audio resultante.

    Returns
    -------
    out: Tensor
        Tensor de forma [T_clipped] donde T_clipped el número de frames
        temporales.
    """
    target_len = int(AUDIO_SAMPLE_RATE * duration)
    L = waveform.size(-1)

    if L > target_len:
        return waveform[..., :target_len]
    else:
        return F.pad(waveform, (0, target_len - L))

def make_image(word: str, x_stride: float = 0.5, y_stride: float = 0.5):
    """Construye una imagen con una palabra dada.

    Parameters
    ----------
    word: str
        Palabra
    x_stride: float
        Posicion en el eje horizontal. 0 = izquierda, 1 = derecha.
    y_stride: float
        Posicion en el eje vertical. 0 = arriba, 1 = abajo.

    Returns
    -------
    x: Tensor
        Tensor de forma [C, H, W] donde C es el número de canales (3), H el
        alto de la imagen (224) y W el ancho (224).
    """
    W = 224
    H = 224

    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype("SF-Mono-Regular.otf", size=60)

    bbox = draw.textbbox((0, 0), word, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # horizontal
    x_pos = int(x_stride * (W - text_w)) - bbox[0]

    # vertical
    y_pos = int(y_stride * (H - text_h)) - bbox[1]

    draw.text((x_pos, y_pos), word, fill="black", font=font)

    x = ToTensor()(img)
    return x

def entrenar_red(net: TrainableModule, data: DataModule, num_epochs: int, phase: int = 1) -> TrainHistory:
    dispositivo = encontrar_dispositivo(silent=True)
    net = mover_a_dispositivo(net, dispositivo)

    optimizer = net.optimizer(phase)

    scheduler = net.scheduler(optimizer, phase)

    train_loader = data.train_loader()
    val_loader = data.val_loader()

    train_history = TrainHistory()

    imprimir_encabezado(net, phase)
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = mover_a_dispositivo(batch, dispositivo)
            optimizer.zero_grad()
            loss = net.training_step(batch, batch_idx, phase)
            running_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        epoch_train_loss = running_loss / len(train_loader)

        net.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = mover_a_dispositivo(batch, dispositivo)
                loss = net.training_step(batch, batch_idx, phase)
                running_loss += loss.item()
        epoch_val_loss = running_loss / len(val_loader)

        scheduler.step()

        train_history.train_losses.append(epoch_train_loss)
        train_history.val_losses.append(epoch_val_loss)

        print(f"Época {epoch+1}/{num_epochs} | " f"train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f}")

    guardar_checkpoint(net, train_history, phase)

    return train_history

def evaluar_red(net: TrainableModule, data: DataModule):
    """Evalúa una red neuronal artificial.

    Parameters
    ----------
    net: TrainableModule
        Red neuronal artificial a evaluar.
    data: DataModule
        Conjunto de datos sobre los cuales evaluar..

    Returns
    -------
    results: TestResults
        Resultado de la evaluación.
    """
    dispositivo= encontrar_dispositivo(silent=True)
    net = mover_a_dispositivo(net, dispositivo)
    test_loader = data.test_loader()
    net.eval()
    running_top1 = 0
    running_top3 = 0
    running_top5 = 0
    num_items = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = mover_a_dispositivo(batch, dispositivo)
            B = data.labels_from_batch(batch).size(0)
            test_results = net.test_step(data, batch)
            running_top1 += test_results.top1
            running_top3 += test_results.top3
            running_top5 += test_results.top5
            num_items += B
    results = TestResults(
        top1=running_top1/num_items,
        top3=running_top3/num_items,
        top5=running_top5/num_items
    )
    return results

def pca(X: Tensor, q: int = 2):
    """Proyecta los datos a sus componentes principales mediante PCA.

    Parameters
    ----------
    X: Tensor
        Tensor de forma [B, D], donde B es el número de ejemplos y D el número
        de dimensiones o características de cada ejemplo.

    q: int
        Número de componentes principales a conservar. Por defecto es 2.

    Returns
    -------
    scores: Tensor
        Tensor de forma [B, q], con la proyección de cada ejemplo
        sobre las componentes principales.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, V = torch.pca_lowrank(X_centered, q=q)
    scores = X_centered @ V[:, :q]
    return scores
