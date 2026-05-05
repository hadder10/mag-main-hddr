"""Security operations used during federated training."""

from .gradient_protection import GradientProtectionConfig, apply_gradient_protection
from .opacus_protection import OpacusProtectionConfig, enable_opacus_protection
from .privacy_backend import (
    PRIVACY_BACKENDS,
    PRIVACY_MANUAL,
    PRIVACY_NONE,
    PRIVACY_OPACUS,
    normalize_privacy_backend,
)

__all__ = [
    "GradientProtectionConfig",
    "OpacusProtectionConfig",
    "PRIVACY_BACKENDS",
    "PRIVACY_MANUAL",
    "PRIVACY_NONE",
    "PRIVACY_OPACUS",
    "apply_gradient_protection",
    "enable_opacus_protection",
    "normalize_privacy_backend",
]
