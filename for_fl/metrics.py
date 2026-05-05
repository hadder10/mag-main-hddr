"""Utilities for saving Flower training metrics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


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


def save_result_metrics(
    result: Any,
    run_config: dict[str, Any],
    output_dir: str | Path = "artifacts/metrics",
) -> Path:
    """Save Flower Result metrics as JSON and CSV files."""

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
    return run_dir
