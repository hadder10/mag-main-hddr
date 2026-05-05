"""FedAvg strategy variant that saves per-client updates for audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }


def _record_to_plain_dict(record: Any | None) -> dict[str, Any]:
    if record is None:
        return {}
    return {str(key): value for key, value in dict(record).items()}


class AuditedFedAvg(FedAvg):
    """FedAvg that stores global and client weights before aggregation."""

    def __init__(
        self,
        *args,
        save_client_updates: bool = True,
        updates_dir: str | Path = "artifacts/updates",
        run_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_client_updates = save_client_updates
        self.updates_dir = Path(updates_dir)
        self.run_config = run_config or {}

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        if self.save_client_updates:
            round_dir = self._round_dir(server_round)
            round_dir.mkdir(parents=True, exist_ok=True)
            self._save_payload(
                round_dir / "global_before.pt",
                state_dict=arrays.to_torch_state_dict(),
                record_type="global_before",
                server_round=server_round,
                train_config=_record_to_plain_dict(config),
            )
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        if self.save_client_updates:
            self._save_client_replies(server_round, replies)
        return super().aggregate_train(server_round, replies)

    def _round_dir(self, server_round: int) -> Path:
        privacy_backend = str(
            self.run_config.get("privacy-backend", "unknown")
        ).replace("/", "_")
        return self.updates_dir / privacy_backend / f"round_{server_round:03d}"

    def _save_client_replies(self, server_round: int, replies: list[Message]) -> None:
        round_dir = self._round_dir(server_round)
        round_dir.mkdir(parents=True, exist_ok=True)

        for index, msg in enumerate(replies):
            if not msg.has_content():
                continue
            arrays = msg.content.get(self.arrayrecord_key)
            if arrays is None:
                continue

            metadata = msg.metadata
            node_id = getattr(metadata, "src_node_id", f"unknown_{index}")
            message_id = str(getattr(metadata, "message_id", index)).replace("/", "_")
            metrics = msg.content.get("metrics")
            filename = f"client_after_node_{node_id}_msg_{message_id}.pt"
            self._save_payload(
                round_dir / filename,
                state_dict=arrays.to_torch_state_dict(),
                record_type="client_after",
                server_round=server_round,
                node_id=node_id,
                message_id=message_id,
                metrics=_record_to_plain_dict(metrics),
            )

    def _save_payload(
        self,
        path: Path,
        state_dict: dict[str, torch.Tensor],
        **metadata: Any,
    ) -> None:
        payload = {
            **metadata,
            "run_config": dict(self.run_config),
            "state_dict": _state_dict_to_cpu(state_dict),
        }
        torch.save(payload, path)
        with (path.with_suffix(".json")).open("w") as handle:
            json.dump(
                {key: value for key, value in payload.items() if key != "state_dict"},
                handle,
                indent=2,
                sort_keys=True,
                default=str,
            )
