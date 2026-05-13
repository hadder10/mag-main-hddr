from __future__ import annotations


PRIVACY_NONE = "none"
PRIVACY_MANUAL = "manual_gradient_protection"
PRIVACY_OPACUS = "opacus"

PRIVACY_BACKENDS = (PRIVACY_NONE, PRIVACY_MANUAL, PRIVACY_OPACUS)

_ALIASES = {
    "": PRIVACY_MANUAL,
    "no": PRIVACY_NONE,
    "off": PRIVACY_NONE,
    "none": PRIVACY_NONE,
    "disable": PRIVACY_NONE,
    "disabled": PRIVACY_NONE,
    "manual": PRIVACY_MANUAL,
    "gradient_protection": PRIVACY_MANUAL,
    "manual_gradient_protection": PRIVACY_MANUAL,
    "opacus": PRIVACY_OPACUS,
}


def normalize_privacy_backend(value: str | None) -> str:
    key = (value or "").lower().strip().replace("-", "_")
    if key not in _ALIASES:
        allowed = ", ".join(PRIVACY_BACKENDS)
        raise ValueError(f"Unknown privacy backend '{value}'. Use one of: {allowed}.")
    return _ALIASES[key]
