from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


PLOT_GROUPS = {
    "accuracy": ("accuracy", "eval_acc"),
    "loss": ("loss", "eval_loss", "train_loss"),
    "f1": ("f1_macro", "f1_weighted", "eval_f1_macro", "eval_f1_weighted"),
}


def _metric_record_to_dict(record: Any) -> dict[str, Any]:
    return {str(key): value for key, value in dict(record).items()}


def _history_to_rows(metric_type: str, history: dict[int, Any]) -> list[dict[str, Any]]:
    rows = []
    for server_round, metric_record in sorted(history.items()):
        row = {"metric_type": metric_type, "round": server_round}
        row.update(_metric_record_to_dict(metric_record))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _plot_metric_group(
    rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    title: str,
    ylabel: str,
    path: Path,
) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False

    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        round_value = row.get("round")
        try:
            server_round = int(round_value)
        except (TypeError, ValueError):
            continue

        metric_type = str(row.get("metric_type", "metric"))
        for metric_name in metric_names:
            value = _as_float(row.get(metric_name))
            if value is None:
                continue
            label = f"{metric_type}.{metric_name}"
            series.setdefault(label, []).append((server_round, value))

    if not series:
        return False

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, points in sorted(series.items()):
        points = sorted(points)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Federated round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _save_metric_plots(
    rows: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[str], str | None]:
    if _load_pyplot() is None:
        return [], "matplotlib is not installed; metric plots were skipped."

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved_plots: list[str] = []
    for group_name, metric_names in PLOT_GROUPS.items():
        plot_path = plots_dir / f"{group_name}.png"
        if _plot_metric_group(
            rows,
            metric_names,
            title=f"{group_name.capitalize()} by federated round",
            ylabel=group_name,
            path=plot_path,
        ):
            saved_plots.append(str(plot_path))

    if saved_plots:
        _plot_overview(rows, plots_dir / "overview.png")
        saved_plots.append(str(plots_dir / "overview.png"))
    return saved_plots, None


def _plot_overview(rows: list[dict[str, Any]], path: Path) -> None:
    plt = _load_pyplot()
    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (group_name, metric_names) in zip(axes, PLOT_GROUPS.items()):
        series: dict[str, list[tuple[int, float]]] = {}
        for row in rows:
            round_value = row.get("round")
            try:
                server_round = int(round_value)
            except (TypeError, ValueError):
                continue

            metric_type = str(row.get("metric_type", "metric"))
            for metric_name in metric_names:
                value = _as_float(row.get(metric_name))
                if value is None:
                    continue
                label = f"{metric_type}.{metric_name}"
                series.setdefault(label, []).append((server_round, value))

        ax.set_title(group_name.capitalize())
        ax.set_ylabel(group_name)
        ax.grid(True, alpha=0.3)
        for label, points in sorted(series.items()):
            points = sorted(points)
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.6,
                label=label,
            )
        if series:
            ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Federated round")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_result_metrics(
    result: Any,
    run_config: dict[str, Any],
    output_dir: str | Path = "artifacts/metrics",
) -> Path:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    privacy_backend = str(run_config.get("privacy-backend", "unknown"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{privacy_backend}"
    run_dir = output_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    histories = {
        "train_clientapp": result.train_metrics_clientapp,
        "evaluate_clientapp": result.evaluate_metrics_clientapp,
        "evaluate_serverapp": result.evaluate_metrics_serverapp,
    }
    payload = {
        "run_name": run_name,
        "run_config": dict(run_config),
        "metrics": {
            name: {
                str(server_round): _metric_record_to_dict(metric_record)
                for server_round, metric_record in sorted(history.items())
            }
            for name, history in histories.items()
        },
    }

    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    all_rows: list[dict[str, Any]] = []
    for metric_type, history in histories.items():
        rows = _history_to_rows(metric_type, history)
        all_rows.extend(rows)
        _write_csv(run_dir / f"{metric_type}.csv", rows)

    _write_csv(run_dir / "metrics.csv", all_rows)
    saved_plots, plot_warning = _save_metric_plots(all_rows, run_dir)
    payload["plots"] = saved_plots
    if plot_warning is not None:
        payload["plot_warning"] = plot_warning
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return run_dir
