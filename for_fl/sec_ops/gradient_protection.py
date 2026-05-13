from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradientProtectionConfig:
    clip_norm: float | None = 1.0
    noise_std: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.clip_norm is not None or self.noise_std > 0


def apply_gradient_protection(
    model: nn.Module,
    config: GradientProtectionConfig,
) -> float | None:

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


def build_clipped_gradient_sum(
    model: nn.Module,
    config: GradientProtectionConfig,
) -> tuple[list[torch.Tensor | None], float]:
    parameters = list(model.parameters())
    total_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=config.clip_norm or float("inf"),
    )
    clipped_grads: list[torch.Tensor | None] = []
    for parameter in parameters:
        if parameter.grad is None:
            clipped_grads.append(None)
            continue
        clipped_grads.append(parameter.grad.detach().clone())
    return clipped_grads, float(total_norm_tensor.item())


def add_clipped_gradient_sum(
    accumulator: list[torch.Tensor | None],
    clipped_grads: list[torch.Tensor | None],
) -> None:
    for index, clipped_grad in enumerate(clipped_grads):
        if clipped_grad is None:
            continue
        if accumulator[index] is None:
            accumulator[index] = torch.zeros_like(clipped_grad)
        accumulator[index].add_(clipped_grad)


def set_noisy_average_gradients(
    model: nn.Module,
    accumulator: list[torch.Tensor | None],
    sample_count: int,
    config: GradientProtectionConfig,
) -> None:
    for parameter, grad_sum in zip(model.parameters(), accumulator):
        if grad_sum is None:
            parameter.grad = None
            continue

        averaged_grad = grad_sum / max(1, sample_count)
        if config.noise_std > 0:
            noise = torch.normal(
                mean=0.0,
                std=config.noise_std,
                size=averaged_grad.shape,
                device=averaged_grad.device,
                dtype=averaged_grad.dtype,
            )
            averaged_grad = averaged_grad + noise
        parameter.grad = averaged_grad
