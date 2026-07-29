import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR

from ..models import DataModule, TestResults, TrainProgramme, TrainableModule
from ..utils import distancia_mel
from .architectures import CoWaverUnconditioned

class MelToMel(nn.Module):
    def __init__(
        self,
        mel_bins: int = 100,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(mel_bins, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_size, mel_bins)

        # Al inicio, la salida es aproximadamente igual al mel de entrada.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, mel):
        """
        mel: [B, mel_bins, T]
        """
        residual = mel
        mel = mel.transpose(1, 2)
        h = self.input_proj(mel)
        h = self.encoder(h)
        delta_mel = self.output_proj(h)
        delta_mel = delta_mel.transpose(1, 2)
        return residual + delta_mel

class CrossAttentionAligner(nn.Module):
    def __init__(self, dim=768, n_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query_seq, kv_seq):
        # query_seq: [B, T, D]  (ej: hablada)
        # kv_seq:    [B, T, D]  (ej: deletreada)
        B, Tq, D = query_seq.shape
        Tk = kv_seq.shape[1]

        Q = self.q_proj(query_seq).view(B, Tq, self.n_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(kv_seq).view(B, Tk, self.n_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(kv_seq).view(B, Tk, self.n_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2,-1)) / (self.head_dim ** 0.5)  # [B, H, Tq, Tk]
        attn = F.softmax(scores, dim=-1)

        aligned = attn @ V  # [B, H, Tq, head_dim]
        aligned = aligned.transpose(1,2).reshape(B, Tq, D)
        aligned = self.out_proj(aligned)

        return aligned, attn.mean(dim=1)

class NonAutoregressiveAlignerV2(nn.Module):
    def __init__(self, dim=768, n_heads=8, n_enc_layers=3, seq_len=49):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads,
                                                 dim_feedforward=dim*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        self.target_queries = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.cross_attn = CrossAttentionAligner(dim=dim, n_heads=n_heads)  # el que ya definimos
        self.refine = nn.Sequential(
            nn.Linear(dim, dim*2), nn.ReLU(), nn.Linear(dim*2, dim)
        )

    def forward(self, src):
        # src: [B, T, D]
        B = src.shape[0]
        memory = self.encoder(src)
        queries = self.target_queries.expand(B, -1, -1)
        aligned, attn_weights = self.cross_attn(queries, memory)  # reusás el bloque anterior
        pred = self.refine(aligned) + aligned  # skip connection
        return pred, attn_weights

def guided_attention_loss(attn, g=0.2):
    # attn: [B, Tq, Tk]
    B, Tq, Tk = attn.shape
    i = torch.arange(Tq, device=attn.device).float().unsqueeze(1) / Tq
    j = torch.arange(Tk, device=attn.device).float().unsqueeze(0) / Tk
    W = 1 - torch.exp(-((i - j) ** 2) / (2 * g ** 2))  # penaliza lejos de diagonal
    return (attn * W).mean()

def entropy_penalty(attn, eps=1e-8):
    ent = -(attn * (attn + eps).log()).sum(dim=-1)  # [B, Tq]
    return -ent.mean()  # negativo: queremos MAXIMIZAR entropía → penalizamos baja entropía

class PhonologicalAwareness(TrainableModule):
    def __init__(self, seq_len: int = 49, mel_bins: int = 80):
        super().__init__(name=f"phonol_awareness_sl{seq_len}_mb{mel_bins}")
        self.mel_bins = mel_bins # dimension (typically, 512)
        self.seq_len = seq_len # time (tipically, 49)
        self.decoder = NonAutoregressiveAlignerV2(dim=mel_bins, n_heads=8)

    def forward(self, x: Tensor) -> tuple:
        spoken_mel, attn_weights = self.decoder(x)
        return spoken_mel, attn_weights

    def training_step(self, batch, batch_idx, phase: int):
        (_, x, y), labels = batch
        y_hat, attn_weights = self(x.squeeze(1))

        loss_ga = guided_attention_loss(attn_weights)

        loss = F.mse_loss(y_hat, y.squeeze(1)) + 1.0 * loss_ga

        return loss

    def test_step(self, data: DataModule, batch: tuple) -> TestResults:
        (_, x, y), targets = batch
        y_hat = self(x)
        prototypes = data.mel_prototypes(y_hat.device)
        distances = distancia_mel(y_hat, prototypes)
        k = min(5, distances.size(1))
        topk = distances.topk(k, dim=1, largest=False).indices
        targets = targets.view(-1, 1)
        top1 = (topk[:, :1] == targets).any(dim=1).sum().item()
        top3 = (topk[:, :min(3, k)] == targets).any(dim=1).sum().item()
        top5 = (topk[:, :k] == targets).any(dim=1).sum().item()
        return TestResults(top1=top1, top3=top3, top5=top5)

    def optimizer(self, phase: int, programme: TrainProgramme) -> Optimizer:
        return AdamW(
            params=self.parameters(),
            lr=programme.epsilon_zero,
            weight_decay=1e-4
        )

    def scheduler(self, optimizer: Optimizer, phase: int, programme: TrainProgramme) -> LRScheduler:
        start_epoch = programme.epochs_before_phase(phase)
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=programme.end_factor,
            total_iters=programme.decay_epochs(),
            last_epoch=start_epoch - 1,
        )


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer


class CrossAttentionAligner(nn.Module):
    def __init__(
        self,
        dim: int = 769,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_seq: Tensor,
        memory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        query_seq:
            Target queries with shape [B, T_target, D].

        memory:
            Source sequence with shape [B, T_source, D].

        Returns
        -------
        aligned:
            Aligned sequence with shape [B, T_target, D].

        attention_weights:
            Per-head attention with shape
            [B, H, T_target, T_source].
        """
        attended, attention_weights = self.attention(
            query=query_seq,
            key=memory,
            value=memory,
            need_weights=True,
            average_attn_weights=False,
        )

        aligned = self.norm(
            query_seq + self.dropout(attended)
        )

        return aligned, attention_weights


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or dim * 4

        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.network(x))


class NonAutoregressiveAligner(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        source_seq_len: int = 49,
        target_seq_len: int = 49,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len

        self.source_positions = nn.Parameter(
            torch.randn(1, source_seq_len, dim) * 0.02
        )

        self.target_queries = nn.Parameter(
            torch.randn(1, target_seq_len, dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(dim),
        )

        self.cross_attention = CrossAttentionAligner(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.refine = FeedForwardBlock(
            dim=dim,
            hidden_dim=dim * 4,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        src:
            Wav2Vec2 internal representations with shape
            [B, T_source, D].

        Returns
        -------
        prediction:
            Predicted target representations with shape
            [B, T_target, D].

        attention_weights:
            Cross-attention weights with shape
            [B, H, T_target, T_source].
        """
        if src.ndim != 3:
            raise ValueError(
                f"Expected src with shape [B, T, D], got {src.shape}"
            )

        batch_size, source_length, dim = src.shape

        if dim != self.dim:
            raise ValueError(
                f"Expected embedding dimension {self.dim}, got {dim}"
            )

        if source_length != self.source_seq_len:
            raise ValueError(
                f"Expected source length {self.source_seq_len}, "
                f"got {source_length}"
            )

        source = src + self.source_positions[:, :source_length]
        memory = self.encoder(source)

        queries = self.target_queries.expand(
            batch_size,
            -1,
            -1,
        )

        aligned, attention_weights = self.cross_attention(
            query_seq=queries,
            memory=memory,
        )

        prediction = self.refine(aligned)

        return prediction, attention_weights


def guided_attention_loss(
    attention: Tensor,
    sigma: float = 0.2,
) -> Tensor:
    """
    Penalize attention far from the normalized diagonal.

    Parameters
    ----------
    attention:
        Shape [B, H, T_target, T_source] or
        [B, T_target, T_source].

    sigma:
        Width of the allowed diagonal region.
    """
    if attention.ndim == 3:
        attention = attention.unsqueeze(1)

    if attention.ndim != 4:
        raise ValueError(
            "Expected attention with shape "
            "[B, H, T_target, T_source]"
        )

    _, _, target_length, source_length = attention.shape
    device = attention.device
    dtype = attention.dtype

    target_positions = torch.linspace(
        0.0,
        1.0,
        target_length,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)

    source_positions = torch.linspace(
        0.0,
        1.0,
        source_length,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)

    distance = target_positions - source_positions

    penalty_mask = 1.0 - torch.exp(
        -(distance.square()) / (2.0 * sigma**2)
    )

    return (attention * penalty_mask).mean()


def attention_entropy_loss(
    attention: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Positive entropy loss.

    Minimizing this encourages sharper attention.
    """
    entropy = -(
        attention * attention.clamp_min(eps).log()
    ).sum(dim=-1)

    return entropy.mean()


def representation_loss(
    prediction: Tensor,
    target: Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> Tensor:
    """
    Compare Wav2Vec2 representations over both values and directions.
    """
    mse = F.mse_loss(prediction, target)

    cosine = 1.0 - F.cosine_similarity(
        prediction,
        target,
        dim=-1,
    ).mean()

    return mse_weight * mse + cosine_weight * cosine


class PhonologicalAwareness(TrainableModule):
    def __init__(
        self,
        seq_len: int = 49,
        embedding_dim: int = 768,
        n_heads: int = 12,
        n_encoder_layers: int = 3,
        guided_attention_weight: float = 0.05,
        cosine_weight: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            name=f"phonological_awareness_sl{seq_len}_d{embedding_dim}"
        )

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.guided_attention_weight = guided_attention_weight
        self.cosine_weight = cosine_weight

        self.decoder = NonAutoregressiveAligner(
            dim=embedding_dim,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            source_seq_len=seq_len,
            target_seq_len=seq_len,
            dropout=dropout,
        )

    @staticmethod
    def normalize_input_shape(x: Tensor) -> Tensor:
        """
        Accept either [B, T, D] or [B, 1, T, D].
        """
        if x.ndim == 4:
            if x.size(1) != 1:
                raise ValueError(
                    f"Expected [B, 1, T, D], got {x.shape}"
                )
            x = x[:, 0]

        if x.ndim != 3:
            raise ValueError(
                f"Expected [B, T, D], got {x.shape}"
            )

        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.normalize_input_shape(x)
        return self.decoder(x)

    def training_step(
        self,
        batch,
        batch_idx: int,
        phase: int,
    ) -> Tensor:
        (_, x, y), labels = batch

        x = self.normalize_input_shape(x)
        y = self.normalize_input_shape(y)

        prediction, attention_weights = self(x)

        loss_representation = representation_loss(
            prediction=prediction,
            target=y,
            mse_weight=1.0,
            cosine_weight=self.cosine_weight,
        )

        loss_guided = guided_attention_loss(
            attention_weights,
            sigma=0.2,
        )

        loss = (
            loss_representation
            + self.guided_attention_weight * loss_guided
        )

        return loss

    def test_step(
        self,
        data: DataModule,
        batch: tuple,
    ) -> TestResults:
        (_, x, y), targets = batch

        prediction, _ = self(x)

        prototypes = data.mel_prototypes(
            prediction.device
        )

        distances = distancia_mel(
            prediction,
            prototypes,
        )

        k = min(5, distances.size(1))

        topk = distances.topk(
            k,
            dim=1,
            largest=False,
        ).indices

        targets = targets.view(-1, 1)

        top1 = (
            topk[:, :1] == targets
        ).any(dim=1).sum().item()

        top3 = (
            topk[:, :min(3, k)] == targets
        ).any(dim=1).sum().item()

        top5 = (
            topk[:, :k] == targets
        ).any(dim=1).sum().item()

        return TestResults(
            top1=top1,
            top3=top3,
            top5=top5,
        )

    def optimizer(
        self,
        phase: int,
        programme: TrainProgramme,
    ) -> Optimizer:
        return AdamW(
            self.parameters(),
            lr=programme.epsilon_zero,
            weight_decay=1e-4,
        )

    def scheduler(
        self,
        optimizer: Optimizer,
        phase: int,
        programme: TrainProgramme,
    ) -> LRScheduler:
        start_epoch = programme.epochs_before_phase(
            phase
        )

        for group in optimizer.param_groups:
            group.setdefault(
                "initial_lr",
                group["lr"],
            )

        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=programme.end_factor,
            total_iters=programme.decay_epochs(),
            last_epoch=start_epoch - 1,
        )
