from __future__ import annotations

import argparse
import os

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

try:
    from .main_task import Net, load_data, settings_from_config
    from .main_task import test as test_fn
    from .main_task import train as train_fn
    from .sec_ops import normalize_privacy_backend
except ImportError:
    from main_task import Net, load_data, settings_from_config
    from main_task import test as test_fn
    from main_task import train as train_fn
    from sec_ops import normalize_privacy_backend


app = ClientApp()


def _privacy_override_from_launch() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--privacy")
    args, _ = parser.parse_known_args()
    value = args.privacy or os.getenv("CLIENT_PRIVACY") or os.getenv("PRIVACY_BACKEND")
    return normalize_privacy_backend(value) if value else None


CLIENT_PRIVACY_OVERRIDE = _privacy_override_from_launch()


def _float_config(config, key: str, default: float) -> float:
    value = config.get(key, default) if config is not None else default
    return float(value)


def _bool_config(config, key: str, default: bool) -> bool:
    value = config.get(key, default) if config is not None else default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _privacy_config(train_config, context: Context) -> str:
    if CLIENT_PRIVACY_OVERRIDE is not None:
        return CLIENT_PRIVACY_OVERRIDE
    return normalize_privacy_backend(
        train_config.get(
            "privacy-backend",
            context.run_config.get("privacy-backend", "manual_gradient_protection"),
        )
    )


def _filter_significant_update(
    initial_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
    threshold: float,
) -> tuple[dict[str, torch.Tensor], int, int]:
    filtered_state: dict[str, torch.Tensor] = {}
    kept_elements = 0
    total_elements = 0

    for name, trained_tensor in trained_state.items():
        initial_tensor = initial_state[name]
        if not torch.is_floating_point(trained_tensor):
            filtered_state[name] = trained_tensor
            continue

        delta = trained_tensor - initial_tensor
        mask = delta.abs() >= threshold
        kept_elements += int(mask.sum().item())
        total_elements += int(mask.numel())
        filtered_state[name] = initial_tensor + delta * mask.to(dtype=delta.dtype)

    return filtered_state, kept_elements, total_elements


@app.train()
def train(msg: Message, context: Context):
    settings = settings_from_config(context.run_config)
    model = Net(num_classes=settings.num_classes)
    initial_state = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(initial_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, settings)

    train_config = msg.content["config"]
    privacy_backend = _privacy_config(train_config, context)
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        _float_config(train_config, "lr", context.run_config["learning-rate"]),
        device,
        grad_clip_norm=_float_config(train_config, "grad-clip-norm", 1.0),
        grad_noise_std=_float_config(train_config, "grad-noise-std", 0.0),
        privacy_backend=privacy_backend,
        opacus_noise_multiplier=_float_config(
            train_config,
            "opacus-noise-multiplier",
            _float_config(train_config, "grad-noise-std", 0.0),
        ),
        opacus_accountant=train_config.get("opacus-accountant", "prv"),
        opacus_delta=_float_config(train_config, "opacus-delta", 1e-5),
        opacus_secure_mode=_bool_config(train_config, "opacus-secure-mode", False),
        opacus_poisson_sampling=_bool_config(
            train_config,
            "opacus-poisson-sampling",
            True,
        ),
        opacus_grad_sample_mode=train_config.get("opacus-grad-sample-mode", "hooks"),
    )

    model_state = model.state_dict()
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    if privacy_backend == "significant_gradient_filter":
        threshold = _float_config(train_config, "significant-threshold", 0.0)
        model_state, kept_elements, total_elements = _filter_significant_update(
            initial_state,
            model_state,
            threshold,
        )
        metrics["significant_threshold"] = threshold
        metrics["significant_kept_elements"] = kept_elements
        metrics["significant_total_elements"] = total_elements
        metrics["significant_kept_ratio"] = kept_elements / max(1, total_elements)

    model_record = ArrayRecord(model_state)
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    settings = settings_from_config(context.run_config)
    model = Net(num_classes=settings.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, settings)

    eval_loss, eval_acc, eval_f1_macro, eval_f1_weighted = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_f1_macro": eval_f1_macro,
        "eval_f1_weighted": eval_f1_weighted,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


if __name__ == "__main__":
    privacy = CLIENT_PRIVACY_OVERRIDE or "server-config"
    print(f"Client privacy backend: {privacy}")
    print("Run this ClientApp through Flower SuperNode; --privacy is used as a client override.")
