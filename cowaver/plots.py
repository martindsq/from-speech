import torch
import matplotlib.pyplot as plt
from .utils import AUDIO_SAMPLE_RATE
from .models import TrainHistory, TestResults

def graficar_waveform(waveform, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3), layout="constrained")
        should_show = True
    else:
        should_show = False

    if waveform.dim() == 2:
        waveform = waveform[0]

    waveform = waveform.detach().cpu()
    time = torch.arange(len(waveform)) / AUDIO_SAMPLE_RATE

    ax.plot(time.numpy(), waveform.numpy())
    ax.set_title("Forma de onda")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)

    if should_show:
        plt.show()

def graficar_mel(mel, ax=None, should_show: bool = False):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3), layout="constrained")

    if mel.dim() == 4:
        mel = mel[0, 0]       # [B,1,mel_bins,T]
    elif mel.dim() == 3:
        mel = mel[0]          # [B,mel_bins,T] o [1,mel_bins,T]
    elif mel.dim() != 2:
        raise ValueError(f"Unexpected mel shape: {mel.shape}")

    mel = torch.expm1(mel).clamp_min(0)
    mel = mel.detach().cpu()

    ax.imshow(
        mel.numpy(),
        cmap="coolwarm",
        aspect="auto",
        origin="lower", interpolation="nearest"
    )

    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Mel-bin Index")
    ax.set_xticks([0, mel.size(1) - 1])
    ax.set_yticks([0, mel.size(0) - 1])

    if should_show:
        plt.show()

def graficar_train_history(train_history: TrainHistory, ax=None):
    should_show = ax is None
    if should_show:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    epochs = range(1, train_history.num_epochs + 1)
    ax.plot(epochs, train_history.train_losses, label="Train")
    ax.plot(epochs, train_history.val_losses, "r", label="Validation")
    ax.set_title("Train vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    if should_show:
        plt.show()

def graficar_test_results(test_results: TestResults, ax=None):
    should_show = ax is None
    if should_show:
        _, ax = plt.subplots(figsize=(10, 3), layout="constrained")
    ax.bar(
        x=["top1", "top3", "top5"],
        height=[test_results.top1, test_results.top3, test_results.top5]
    )
    ax.set_title("Test Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    if should_show:
        plt.show()

def graficar_embedding(activation, ax=None, vmin=-3, vmax=3):
    should_show = ax is None
    if should_show:
        _, ax = plt.subplots(figsize=(9, 3), layout="constrained")

    if activation.dim() == 4:
        activation = activation[0, 0]
    elif activation.dim() == 3:
        activation = activation[0]
    elif activation.dim() != 2:
        raise ValueError(f"Unexpected activation shape: {activation.shape}")

    activation = activation.detach().cpu()

    im = ax.imshow(
        activation.numpy(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_title("Activación")
    ax.set_xlabel("Dimensión")
    ax.set_ylabel("Ventana")
    ax.set_xticks([0, activation.size(1) - 1])
    ax.set_yticks([0, activation.size(0) - 1])

    if should_show:
        plt.colorbar(im, ax=ax)
        plt.show()

    return im
