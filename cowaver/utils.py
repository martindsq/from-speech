import torch
import torchaudio
from torch import Tensor, nn, device
import torch.nn.functional as F
from torchaudio.transforms import Resample, InverseMelScale, GriffinLim
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
import shutil
import tarfile
import ipywidgets as widgets
import unicodedata
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable
from torchvision.transforms import ToTensor
from torchaudio.transforms import MelSpectrogram
from .checkpoints import imprimir_encabezado, guardar_checkpoint
from .models import DataModule, TestResults, TrainHistory, TrainProgramme, TrainableModule

AUDIO_SAMPLE_RATE = 16_000
FONT_PATH = font_manager.findfont("DejaVu Sans Mono")
REEMPLAZOS_ACENTOS = str.maketrans({
    "á": "a",
    "é": "e",
    "í": "i",
    "ó": "o",
    "ú": "u",
    "ü": "u",
    "Á": "a",
    "É": "e",
    "Í": "i",
    "Ó": "o",
    "Ú": "u",
    "Ü": "u",
})

def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto.lower())
    return texto.translate(REEMPLAZOS_ACENTOS)

def listar_clases(carpeta: Path) -> list[str]:
    """Lista las clases contenidas en una carpeta.

    Parameters
    ----------
    carpeta: Path
        Carpeta que contiene una subcarpeta por cada clase.

    Returns
    -------
    clases: list[str]
        Lista ordenada con los nombres de las clases encontradas.
    """
    clases = [
        path.name for path in sorted(carpeta.iterdir())
        if not path.name.startswith(".") and path.is_dir()
    ]
    return clases

def seleccionar_clases(base_path: Path, max_classes: int, seed: int = 42) -> list[str]:
    """Selecciona un subconjunto reproducible de clases y lo devuelve ordenado.

    Parameters
    ----------
    base_path: Path
        Carpeta base del dataset.
    max_classes: int
        Numero maximo de clases a seleccionar.
    seed: int
        Semilla para seleccionar las clases de forma reproducible.

    Returns
    -------
    clases: list[str]
        Lista ordenada con las clases seleccionadas.
    """
    classes = listar_clases(base_path / "train")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(classes), generator=generator).tolist()
    clases = sorted(classes[index] for index in permutation[:max_classes])
    return clases

def separar_clases(classes: list[str], fraction: float = 0.1, seed: int = 42):
    """Separa clases de train/test de forma reproducible.

    Parameters
    ----------
    classes: list[str]
        Lista de clases a separar.
    fraction: float
        Fraccion de clases a usar para test.
    seed: int
        Semilla para separar las clases de forma reproducible.

    Returns
    -------
    train: list[str]
        Lista ordenada con las clases de train.
    test: list[str]
        Lista ordenada con las clases de test.
    """
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

def construir_vocabulario_caracteres(datasets: list[tuple[Path, list[str] | None]]) -> dict[str, int]:
    """Construye un vocabulario de caracteres a partir de clases de datasets.

    Parameters
    ----------
    datasets: list[tuple[Path, list[str] | None]]
        Lista de pares con la carpeta base de un dataset y las clases a usar.
        Si las clases son None, se usan todas las clases de train y test.

    Returns
    -------
    vocabulario: dict[str, int]
        Diccionario que asigna un indice a cada caracter normalizado.
    """
    caracteres = set()
    for base_dir, classes in datasets:
        if classes is None:
            for split in ("train", "test"):
                for clase in listar_clases(base_dir / split):
                    caracteres.update(normalizar_texto(clase))
        else:
            for clase in classes:
                caracteres.update(normalizar_texto(clase))
    vocabulario = {
        caracter: indice + 1
        for indice, caracter in enumerate(sorted(caracteres))
    }
    return vocabulario

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
    if not silent:
        print("Buscando dispositivo", end="... ")
    if torch.cuda.is_available():
        dispositivo = device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dispositivo = device("mps")
    else:
        dispositivo = device("cpu")
    if not silent:
        print(dispositivo.type.upper())
    return dispositivo

def mover_a_dispositivo(x: Tensor | nn.Module | tuple | list | Mapping, dispositivo: device | None = None):
    """Mueve tensores o módulos a un dispositivo apropiado para entrenar o evaluar.

    Parameters
    ----------
    x: Tensor | nn.Module | tuple | list | Mapping
        Un tensor cualquiera, un módulo, o una colección cuyos elementos sean
        tensores o módulos.
    dispositivo: device | None
        Dispositivo apropiado. Si es None, entonces se encuentra uno.

    Returns
    -------
    x: Tensor | nn.Module | tuple | list | Mapping
        Igual a x pasado como parámetro, pero ubicado en el dispositivo
        apropiado.
    """
    if dispositivo is None:
        dispositivo = encontrar_dispositivo(silent=True)
    if torch.is_tensor(x) or isinstance(x, nn.Module):
        return x.to(dispositivo)
    if isinstance(x, Mapping):
        return type(x)((k, mover_a_dispositivo(v, dispositivo)) for k, v in x.items())
    if isinstance(x, tuple):
        return tuple(mover_a_dispositivo(v, dispositivo) for v in x)
    if isinstance(x, list):
        return [mover_a_dispositivo(v, dispositivo) for v in x]
    return x

def descomprimir_archivo(archivo: Path, carpeta: Path) -> Path:
    ruta = carpeta / Path(archivo.stem).stem
    print(f"Descomprimiendo {archivo} en {ruta}", end="... ")
    if not ruta.exists():
        with tarfile.open(archivo, mode="r:xz") as tar:
            tar.extractall(path=carpeta)
        print("OK")
    else:
        print("YA EXISTE")
    return ruta

def borrar_carpeta(carpeta: Path):
    print(f"Borrando {carpeta}", end="... ")
    shutil.rmtree(carpeta)
    print("OK")

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

def explorar_red(net: TrainableModule, data: DataModule, f: Callable[[TrainableModule, Any], Any]):
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

def extract_mel(waveform: Tensor, mel_bins: int = 80):
    """Calcula un espectrograma Mel.

    Parameters
    ----------
    waveform: Tensor
        Tensor de forma [B, T] donde B es el tamaño del batch y T el número de
        frames temporales. Se puede omitir B.

    Returns
    -------
    mel: Tensor
        Espectrograma Mel comprimido de forma [B, mel_bins, T]. B es igual a 1
        si se omitió en waveform.
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

    return torch.log1p(mel)

def extraer_waveform(mel: Tensor):
    """Construye un waveform a partir de un espectrograma Mel.

    Parameters
    ----------
    mel: Tensor
        Tensor de forma [B, mel_bins, T] donde T el número de frames temporales.

    Returns
    -------
    waveform: Tensor
        Tensor de forma [B, 1, T] con el waveform extraído.
    """

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)

    mel = torch.expm1(mel).clamp_min(0)

    n_fft = 2048
    hop_length = int(round(AUDIO_SAMPLE_RATE / 49))

    n_mels = mel.size(-2)

    inverse_mel = InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=AUDIO_SAMPLE_RATE,
        f_min=0.0,
        f_max=AUDIO_SAMPLE_RATE / 2,
    ).to(mel.device)

    griffin_lim = GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=64,
    ).to(mel.device)

    spectrogram = inverse_mel(mel)
    waveform = griffin_lim(spectrogram)

    return waveform

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
    word = unicodedata.normalize("NFC", word)

    img = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(FONT_PATH, size=60)

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

def entrenar_red(net: TrainableModule, data: DataModule, programme: TrainProgramme, phase: int = 1, dispositivo: device | None = None, checkpoints_folder: Path | None = None) -> TrainHistory:
    if dispositivo is None:
        dispositivo = encontrar_dispositivo(silent=True)
    net = mover_a_dispositivo(net, dispositivo)

    optimizer = net.optimizer(phase, programme)

    scheduler = net.scheduler(optimizer, phase, programme)
    num_epochs = programme.epochs_for_phase(phase)

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

        print(f"Epoch {epoch+1}/{num_epochs} | " f"train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f}")

    if checkpoints_folder is not None:
        guardar_checkpoint(net, train_history, programme, phase, checkpoints_folder)

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

def distancia_mel(mel_a: torch.Tensor, mel_b: torch.Tensor) -> torch.Tensor:
    """Calcula una distancia DTW entre espectrogramas mel.

    Parameters
    ----------
    mel_a: Tensor
        Tensor de forma [mel_bins, seq_len] o [B, mel_bins, seq_len],
        correspondiente al primer espectrograma mel o a un batch.

    mel_b: Tensor
        Tensor de forma [mel_bins, seq_len] o [P, mel_bins, seq_len],
        correspondiente al segundo espectrograma mel o a un conjunto de
        espectrogramas.

    Returns
    -------
    distances: Tensor
        Tensor de forma [B, P], donde distances[b, p] contiene la distancia
        DTW normalizada entre mel_a[b] y mel_b[p]. Si alguna entrada era 2D,
        se interpreta como un batch de tamaño 1.
    """

    if mel_a.dim() == 2:
        mel_a = mel_a.unsqueeze(0)
    if mel_b.dim() == 2:
        mel_b = mel_b.unsqueeze(0)

    x = mel_a.transpose(1, 2)
    y = mel_b.transpose(1, 2)

    _, T1, D1 = x.shape
    _, T2, D2 = y.shape

    if D1 != D2:
        raise ValueError(f"Dimensiones incompatibles: {x.shape} vs {y.shape}")
    
    x = F.normalize(x, dim=2)
    y = F.normalize(y, dim=2)
    cost = 1 - torch.einsum("btd,psd->bpts", x, y)

    dtw = torch.full(
        (x.size(0), y.size(0), T1 + 1, T2 + 1),
        float("inf"),
        device=mel_a.device,
        dtype=mel_a.dtype,
    )

    dtw[:, :, 0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            dtw[:, :, i, j] = cost[:, :, i - 1, j - 1] + torch.minimum(
                torch.minimum(dtw[:, :, i - 1, j], dtw[:, :, i, j - 1]),
                dtw[:, :, i - 1, j - 1],
            )

    distances = dtw[:, :, T1, T2] / (T1 + T2)
    return distances
