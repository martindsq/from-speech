from .conditioned import CoWaverConditioned
from .dual_route import CoWaverDualRoute
from .adapters import (
    ConvolutionalTemporalAdapter,
    IdentityAdapter,
    IdentityTemporalAdapter,
    RecurrentAdapter,
    RecurrentTemporalAdapter,
    TEMPORAL_ADAPTER_REGISTRY,
    TemporalAdapter,
    TransformerAdapter,
    TransformerTemporalAdapter,
    build_temporal_adapter,
)
from .cornet import CORblock_Z, CORnet_Z, Flatten, Identity
from .common import CTCHead, ResidualTemporalBlock, unpack_batch
from .decoders import (
    DECODER_REGISTRY,
    ConvolutionalMelDecoder,
    HorizontalFeaturesToMel,
    RecurrentMelDecoder,
    Seq2SeqMelDecoder,
    TransformerMelDecoder,
    build_decoder,
)
from .encoders import AvgPooledITEncoder, ImageToHorizontalFeatures
from .unconditioned import CoWaverUnconditioned

CoWaver = CoWaverUnconditioned

MODEL_REGISTRY = {
    "unconditioned": CoWaverUnconditioned,
    "dual-route": CoWaverDualRoute,
    "conditioned": CoWaverConditioned,
}


def build_model(architecture: str = "unconditioned", **kwargs):
    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown architecture '{architecture}'. Options: {options}") from exc
    return model_cls(**kwargs)


__all__ = [
    "CORblock_Z",
    "CORnet_Z",
    "CoWaver",
    "CoWaverConditioned",
    "CoWaverDualRoute",
    "CoWaverUnconditioned",
    "ConvolutionalMelDecoder",
    "ConvolutionalTemporalAdapter",
    "CTCHead",
    "DECODER_REGISTRY",
    "Flatten",
    "HorizontalFeaturesToMel",
    "Identity",
    "IdentityAdapter",
    "IdentityTemporalAdapter",
    "AvgPooledITEncoder",
    "ImageToHorizontalFeatures",
    "MODEL_REGISTRY",
    "RecurrentMelDecoder",
    "RecurrentAdapter",
    "RecurrentTemporalAdapter",
    "ResidualTemporalBlock",
    "Seq2SeqMelDecoder",
    "TEMPORAL_ADAPTER_REGISTRY",
    "TemporalAdapter",
    "TransformerAdapter",
    "TransformerMelDecoder",
    "TransformerTemporalAdapter",
    "build_decoder",
    "build_model",
    "build_temporal_adapter",
    "unpack_batch",
]
