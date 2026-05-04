"""Flower server app for CIFAR-100 + Google Landmarks v2."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

try:
    from .main_task import Net, load_centralized_dataset, settings_from_config, test
except ImportError:
    from main_task import Net, load_centralized_dataset, settings_from_config, test


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the central Flower server."""

    fraction_train: float = context.run_config.get("fraction-train", 0.2)
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    settings = settings_from_config(context.run_config)

    global_model = Net(num_classes=settings.num_classes)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=int(context.run_config.get("min-train-nodes", 2)),
        min_evaluate_nodes=int(context.run_config.get("min-evaluate-nodes", 2)),
        min_available_nodes=int(context.run_config.get("min-available-nodes", 2)),
    )

    # grad-noise-std и grad-clip-norm передаются каждому клиенту.
    # На клиентской стороне они включают DP-SGD-подобное добавление шума
    # к градиентам перед локальным optimizer.step(), чтобы замаскировать
    # индивидуальный вклад данных при последующей федеративной агрегации.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "grad-noise-std": float(context.run_config.get("grad-noise-std", 0.0)),
                "grad-clip-norm": float(context.run_config.get("grad-clip-norm", 1.0)),
            }
        ),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, settings
        ),
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, settings=None
) -> MetricRecord:
    """Evaluate the global model on central validation data."""

    model_state = arrays.to_torch_state_dict()
    classifier_weight = model_state["classifier.weight"]
    model = Net(num_classes=classifier_weight.shape[0])
    model.load_state_dict(model_state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    settings = settings or settings_from_config(None)
    test_dataloader = load_centralized_dataset(settings=settings)
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
