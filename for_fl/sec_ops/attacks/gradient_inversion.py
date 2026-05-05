"""Simulate a gradient inversion attack on saved federated client updates.

This script is intended for defensive evaluation of the gradient protection
backends in this project. It reconstructs a dummy input by matching the
gradients implied by a saved client update.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

try:
    from ...main_task import Net
except ImportError:
    from for_fl.main_task import Net


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    if isinstance(payload, dict):
        return {"state_dict": payload}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _find_client_file(round_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = sorted(round_dir.glob("client_after_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No client_after_*.pt files found in {round_dir}")
    return candidates[0]


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise ValueError("Cannot infer num_classes: classifier.weight is missing.")
    return int(classifier_weight.shape[0])


def _infer_model_kwargs(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    kwargs = {"num_classes": _infer_num_classes(state_dict)}
    first_conv = state_dict.get("features.0.weight")
    classifier_weight = state_dict.get("classifier.weight")
    if first_conv is not None:
        kwargs["width"] = int(first_conv.shape[0])
    if classifier_weight is not None:
        kwargs["embedding_dim"] = int(classifier_weight.shape[1])
    return kwargs


def _infer_label(target_grads: dict[str, torch.Tensor]) -> int | None:
    bias_grad = target_grads.get("classifier.bias")
    if bias_grad is None:
        return None
    return int(torch.argmin(bias_grad).item())


def _select_target_gradients(
    model: torch.nn.Module,
    global_state: dict[str, torch.Tensor],
    client_state: dict[str, torch.Tensor],
    learning_rate: float,
    param_regex: str,
    max_param_elements: int,
    device: torch.device,
) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, torch.Tensor]]:
    pattern = re.compile(param_regex)
    selected_params: list[tuple[str, torch.nn.Parameter]] = []
    target_grads: dict[str, torch.Tensor] = {}
    total_elements = 0

    for name, parameter in model.named_parameters():
        if not pattern.search(name):
            continue
        if name not in global_state or name not in client_state:
            continue
        if not torch.is_floating_point(global_state[name]):
            continue
        num_elements = int(parameter.numel())
        if total_elements + num_elements > max_param_elements:
            continue

        target = (global_state[name] - client_state[name]) / learning_rate
        selected_params.append((name, parameter))
        target_grads[name] = target.detach().to(device)
        total_elements += num_elements

    if not selected_params:
        raise ValueError(
            "No parameters selected for attack. Relax --param-regex or "
            "increase --max-param-elements."
        )
    return selected_params, target_grads


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


def _total_variation(image: torch.Tensor) -> torch.Tensor:
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical


def _gradient_matching_loss(
    dummy_grads: tuple[torch.Tensor | None, ...],
    selected_names: list[str],
    target_grads: dict[str, torch.Tensor],
) -> torch.Tensor:
    loss = None
    for name, dummy_grad in zip(selected_names, dummy_grads):
        if dummy_grad is None:
            continue
        target = target_grads[name]
        denom = target.pow(2).mean().clamp_min(1e-12)
        term = (dummy_grad - target).pow(2).mean() / denom
        loss = term if loss is None else loss + term
    if loss is None:
        raise RuntimeError("All selected gradients were None during attack.")
    return loss


def run_attack(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    round_dir = Path(args.round_dir)
    global_file = round_dir / "global_before.pt"
    client_file = _find_client_file(round_dir, args.client_file)

    global_payload = _load_payload(global_file, device)
    client_payload = _load_payload(client_file, device)
    global_state = global_payload["state_dict"]
    client_state = client_payload["state_dict"]
    model_kwargs = _infer_model_kwargs(global_state)
    if args.num_classes is not None:
        model_kwargs["num_classes"] = args.num_classes

    model = Net(**model_kwargs).to(device)
    model.load_state_dict(global_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    selected_params, target_grads = _select_target_gradients(
        model=model,
        global_state=global_state,
        client_state=client_state,
        learning_rate=args.learning_rate,
        param_regex=args.param_regex,
        max_param_elements=args.max_param_elements,
        device=device,
    )
    selected_names = [name for name, _ in selected_params]
    selected_tensors = [parameter for _, parameter in selected_params]

    target_label = args.target_label
    if target_label is None:
        target_label = _infer_label(target_grads)
    if target_label is None:
        raise ValueError("Cannot infer target label. Pass --target-label explicitly.")

    run_name = time.strftime("%Y%m%d-%H%M%S-gradient-inversion")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_image = torch.rand(
        1,
        3,
        args.image_size,
        args.image_size,
        device=device,
        requires_grad=True,
    )
    target = torch.tensor([target_label], device=device, dtype=torch.long)
    optimizer = torch.optim.Adam([dummy_image], lr=args.attack_lr)
    rows: list[dict[str, float | int]] = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        logits = model(_normalize_image(dummy_image))
        classification_loss = F.cross_entropy(logits, target)
        dummy_grads = torch.autograd.grad(
            classification_loss,
            selected_tensors,
            create_graph=True,
            allow_unused=True,
        )
        match_loss = _gradient_matching_loss(dummy_grads, selected_names, target_grads)
        tv_loss = _total_variation(dummy_image)
        image_l2 = dummy_image.pow(2).mean()
        objective = match_loss + args.tv_weight * tv_loss + args.l2_weight * image_l2

        objective.backward()
        optimizer.step()
        with torch.no_grad():
            dummy_image.clamp_(0.0, 1.0)

        if step % args.log_every == 0 or step == args.steps - 1:
            rows.append(
                {
                    "step": step,
                    "objective": float(objective.detach().cpu()),
                    "gradient_match_loss": float(match_loss.detach().cpu()),
                    "tv_loss": float(tv_loss.detach().cpu()),
                    "image_l2": float(image_l2.detach().cpu()),
                }
            )

    save_image(dummy_image.detach().cpu(), output_dir / "reconstructed.png")
    with (output_dir / "losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "round_dir": str(round_dir),
        "global_file": str(global_file),
        "client_file": str(client_file),
        "target_label": target_label,
        "model_kwargs": model_kwargs,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "steps": args.steps,
        "attack_lr": args.attack_lr,
        "param_regex": args.param_regex,
        "selected_parameters": selected_names,
        "client_metadata": {
            key: value
            for key, value in client_payload.items()
            if key != "state_dict"
        },
        "global_metadata": {
            key: value
            for key, value in global_payload.items()
            if key != "state_dict"
        },
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gradient inversion against saved FL client updates."
    )
    parser.add_argument("--round-dir", required=True, help="Directory with one round.")
    parser.add_argument("--client-file", help="Specific client_after_*.pt file.")
    parser.add_argument("--output-dir", default="artifacts/attacks")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--target-label", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--attack-lr", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=1e-4)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--param-regex",
        default=r"features|embedding|classifier\.bias",
        help="Regex for model parameters used in gradient matching.",
    )
    parser.add_argument(
        "--max-param-elements",
        type=int,
        default=2_000_000,
        help="Maximum selected parameter elements to keep memory bounded.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    output_dir = run_attack(parse_args())
    print(f"Saved gradient inversion results to {output_dir}")


if __name__ == "__main__":
    main()
