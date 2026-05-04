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


@app.train()
def train(msg: Message, context: Context):
    """Train the model on one local CIFAR-100 + GLDv2 shard."""

    settings = settings_from_config(context.run_config)
    model = Net(num_classes=settings.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, settings)

    train_config = msg.content["config"]
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        _float_config(train_config, "lr", context.run_config["learning-rate"]),
        device,
        grad_clip_norm=_float_config(train_config, "grad-clip-norm", 1.0),
        grad_noise_std=_float_config(train_config, "grad-noise-std", 0.0),
        privacy_backend=_privacy_config(train_config, context),
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

    # Здесь в федеративный цикл возвращаются веса после локального шага.
    # Если включен grad-noise-std, эти веса уже получены из зашумленных градиентов,
    # что имитирует приватное "шифрование" вклада клиента перед агрегацией.
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on one local validation shard."""

    settings = settings_from_config(context.run_config)
    model = Net(num_classes=settings.num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size, settings)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


if __name__ == "__main__":
    privacy = CLIENT_PRIVACY_OVERRIDE or "server-config"
    print(f"Client privacy backend: {privacy}")
    print("Run this ClientApp through Flower SuperNode; --privacy is used as a client override.")
