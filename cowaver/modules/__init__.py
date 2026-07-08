from .architectures import (
    CoWaver,
    CoWaverConditioned,
    CoWaverDualRoute,
    CoWaverUnconditioned
)
from .phonological_awareness import PhonologicalAwareness
from .adapters import ADAPTER_REGISTRY
from .decoders import DECODER_REGISTRY

ARCHITECTURE_REGISTRY = {
    "unconditioned": CoWaverUnconditioned,
    "dual-route": CoWaverDualRoute,
    "conditioned": CoWaverConditioned,
}


def build_model(architecture: str = "unconditioned", **kwargs):
    try:
        model_cls = ARCHITECTURE_REGISTRY[architecture]
    except KeyError as exc:
        options = ", ".join(sorted(ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unknown architecture '{architecture}'. Options: {options}") from exc
    return model_cls(**kwargs)


__all__ = [
    "ADAPTER_REGISTRY",
    "ARCHITECTURE_REGISTRY",
    "CoWaver",
    "CoWaverConditioned",
    "CoWaverDualRoute",
    "CoWaverUnconditioned",
    "DECODER_REGISTRY",
    "build_model",
    "PhonologicalAwareness"
]
