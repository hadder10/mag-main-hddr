"""Gradient protection utilities for federated learning clients."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradientProtectionConfig:
    """Configuration for clipping gradients and adding Gaussian noise."""

    clip_norm: float | None = 1.0
    noise_std: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.clip_norm is not None or self.noise_std > 0


def apply_gradient_protection(
    model: nn.Module,
    config: GradientProtectionConfig,
) -> float | None:
    """Clip gradients and add Gaussian noise before the optimizer step.

    In federated learning, this function is called on each client after
    ``loss.backward()`` and before ``optimizer.step()``. Clipping bounds the
    influence of one batch/client update, and Gaussian noise masks the exact
    gradient signal before the resulting local weights are sent to the server.
    This is a DP-SGD-style protection layer, not a replacement for secure
    aggregation or cryptographic encryption.
    """

    total_norm = None
    if config.clip_norm is not None:
        total_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.clip_norm,
        )
        total_norm = float(total_norm_tensor.item())

    if config.noise_std > 0:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            noise = torch.normal(
                mean=0.0,
                std=config.noise_std,
                size=parameter.grad.shape,
                device=parameter.grad.device,
                dtype=parameter.grad.dtype,
            )
            parameter.grad.add_(noise)

    return total_norm
