from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class OpacusProtectionConfig:
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    accountant: str = "prv"
    secure_mode: bool = False
    poisson_sampling: bool = True
    grad_sample_mode: str = "hooks"
    delta: float = 1e-5


@dataclass
class OpacusProtectionState:
    model: nn.Module
    optimizer: Optimizer
    trainloader: DataLoader
    privacy_engine: Any
    hooks: Any | None = None

    def get_epsilon(self, delta: float) -> float | None:
        try:
            return float(self.privacy_engine.get_epsilon(delta))
        except Exception:
            return None

    def cleanup(self) -> None:
        if self.hooks is not None and hasattr(self.hooks, "cleanup"):
            self.hooks.cleanup()


def enable_opacus_protection(
    model: nn.Module,
    optimizer: Optimizer,
    trainloader: DataLoader,
    config: OpacusProtectionConfig,
) -> OpacusProtectionState:

    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise ImportError(
        ) from exc

    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=True)

    privacy_engine = PrivacyEngine(
        accountant=config.accountant,
        secure_mode=config.secure_mode,
    )
    hooks, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.max_grad_norm,
        poisson_sampling=config.poisson_sampling,
        grad_sample_mode=config.grad_sample_mode,
        wrap_model=False,
    )
    return OpacusProtectionState(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        privacy_engine=privacy_engine,
        hooks=hooks,
    )
