"""Security operations used during federated training."""

from .gradient_protection import (
    GradientProtectionConfig,
    add_clipped_gradient_sum,
    apply_gradient_protection,
    build_clipped_gradient_sum,
    set_noisy_average_gradients,
)
from .opacus_protection import OpacusProtectionConfig, enable_opacus_protection
from .privacy_backend import (
    PRIVACY_BACKENDS,
    PRIVACY_MANUAL,
    PRIVACY_NONE,
    PRIVACY_OPACUS,
    PRIVACY_SIGNIFICANT,
    normalize_privacy_backend,
)

__all__ = [
    "GradientProtectionConfig",
    "OpacusProtectionConfig",
    "PRIVACY_BACKENDS",
    "PRIVACY_MANUAL",
    "PRIVACY_NONE",
    "PRIVACY_OPACUS",
    "PRIVACY_SIGNIFICANT",
    "add_clipped_gradient_sum",
    "apply_gradient_protection",
    "build_clipped_gradient_sum",
    "enable_opacus_protection",
    "normalize_privacy_backend",
    "set_noisy_average_gradients",
]
