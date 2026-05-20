from .conditioned import CoWaverConditioned
from .dual_route import CoWaverDualRoute
from .recurrent import CoWaverRecurrent
from .cornet import CORblock_Z, CORnet_Z, Flatten, Identity
from .convolutional import (
    CoWaverConvolutional,
    HorizontalFeaturesToMel,
    ImageToHorizontalFeatures,
    ResidualTemporalBlock,
    TemporalAdapter,
)
from .transformer import CoWaverTransformer

CoWaver = CoWaverConvolutional

MODEL_REGISTRY = {
    "convolutional": CoWaverConvolutional,
    "transformer": CoWaverTransformer,
    "recurrent": CoWaverRecurrent,
    "dual-route": CoWaverDualRoute,
    "conditioned": CoWaverConditioned,
}


def build_model(architecture: str = "convolutional", **kwargs):
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
    "CoWaverRecurrent",
    "CoWaverConvolutional",
    "CoWaverTransformer",
    "Flatten",
    "HorizontalFeaturesToMel",
    "Identity",
    "ImageToHorizontalFeatures",
    "MODEL_REGISTRY",
    "ResidualTemporalBlock",
    "TemporalAdapter",
    "build_model",
]
