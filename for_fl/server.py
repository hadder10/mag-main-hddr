from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

try:
    from .audit_strategy import AuditedFedAvg
    from .metrics import save_result_metrics
    from .main_task import Net, load_centralized_dataset, settings_from_config, test
except ImportError:
    from audit_strategy import AuditedFedAvg
    from metrics import save_result_metrics
    from main_task import Net, load_centralized_dataset, settings_from_config, test


app = ServerApp()


def _bool_config(config, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config.get("fraction-train", 0.2)
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    settings = settings_from_config(context.run_config)
    central_evaluate = _bool_config(context.run_config, "central-evaluate", False)

    global_model = Net(num_classes=settings.num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = AuditedFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=int(context.run_config.get("min-train-nodes", 2)),
        min_evaluate_nodes=int(context.run_config.get("min-evaluate-nodes", 2)),
        min_available_nodes=int(context.run_config.get("min-available-nodes", 2)),
        save_client_updates=_bool_config(context.run_config, "save-client-updates", True),
        updates_dir=context.run_config.get("updates-dir", "artifacts/updates"),
        run_config=context.run_config,
    )
    evaluate_fn = None
    if central_evaluate:
        evaluate_fn = lambda server_round, arrays: global_evaluate(
            server_round,
            arrays,
            settings,
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "privacy-backend": context.run_config.get(
                    "privacy-backend",
                    "manual_gradient_protection",
                ),
                "grad-noise-std": float(context.run_config.get("grad-noise-std", 0.0)),
                "grad-clip-norm": float(context.run_config.get("grad-clip-norm", 1.0)),
                "opacus-noise-multiplier": float(
                    context.run_config.get(
                        "opacus-noise-multiplier",
                        context.run_config.get("grad-noise-std", 0.0),
                    )
                ),
                "opacus-accountant": context.run_config.get("opacus-accountant", "prv"),
                "opacus-delta": float(context.run_config.get("opacus-delta", 1e-5)),
                "opacus-secure-mode": _bool_config(
                    context.run_config, "opacus-secure-mode", False
                ),
                "opacus-poisson-sampling": _bool_config(
                    context.run_config, "opacus-poisson-sampling", True
                ),
                "opacus-grad-sample-mode": context.run_config.get(
                    "opacus-grad-sample-mode",
                    "hooks",
                ),
                "significant-threshold": float(
                    context.run_config.get("significant-threshold", 0.0)
                ),
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    model_dir = Path(context.run_config.get("model-dir", "artifacts/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "final_model.pt"
    torch.save(state_dict, final_model_path)
    print(f"Saved final model to {final_model_path}")
    metrics_dir = save_result_metrics(
        result,
        context.run_config,
        output_dir=context.run_config.get("metrics-dir", "artifacts/metrics"),
    )
    print(f"Saved metrics to {metrics_dir}")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, settings=None
) -> MetricRecord:
    model_state = arrays.to_torch_state_dict()
    classifier_weight = model_state["classifier.weight"]
    model = Net(num_classes=classifier_weight.shape[0])
    model.load_state_dict(model_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    settings = settings or settings_from_config(None)
    test_dataloader = load_centralized_dataset(settings=settings)
    test_loss, test_acc, test_f1_macro, test_f1_weighted = test(
        model,
        test_dataloader,
        device,
    )

    return MetricRecord(
        {
            "accuracy": test_acc,
            "loss": test_loss,
            "f1_macro": test_f1_macro,
            "f1_weighted": test_f1_weighted,
        }
    )
