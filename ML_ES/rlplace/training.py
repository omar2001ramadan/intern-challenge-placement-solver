"""Logging, checkpointing, and adaptive LR utilities."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Persist a torch payload atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target)


class JsonlLogger:
    """Append-only JSONL logger for long-running training jobs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        serializable: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (float, int, str, bool)) or value is None:
                serializable[key] = value
            else:
                serializable[key] = str(value)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable) + "\n")


class TerminationMonitor:
    """Track SIGINT and SIGTERM so training can checkpoint cleanly."""

    def __init__(self) -> None:
        self.should_stop = False
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        def _handler(signum, frame):  # type: ignore[unused-argument]
            self.should_stop = True

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        self._installed = True


class LossAdaptiveLRScheduler:
    """Increase LR when monitored loss improves and decrease it on regressions."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        min_lr: float,
        max_lr: float,
        increase_factor: float,
        decrease_factor: float,
        tolerance: float = 1e-4,
    ) -> None:
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.tolerance = tolerance
        self.previous_metric: float | None = None

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self, metric: float) -> float:
        if self.previous_metric is None:
            self.previous_metric = metric
            return self.current_lr()

        current_lr = self.current_lr()
        if metric < self.previous_metric * (1.0 - self.tolerance):
            new_lr = min(current_lr * self.increase_factor, self.max_lr)
        elif metric > self.previous_metric * (1.0 + self.tolerance):
            new_lr = max(current_lr * self.decrease_factor, self.min_lr)
        else:
            new_lr = current_lr

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        self.previous_metric = metric
        return new_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "increase_factor": self.increase_factor,
            "decrease_factor": self.decrease_factor,
            "tolerance": self.tolerance,
            "previous_metric": self.previous_metric,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.min_lr = float(state_dict["min_lr"])
        self.max_lr = float(state_dict["max_lr"])
        self.increase_factor = float(state_dict["increase_factor"])
        self.decrease_factor = float(state_dict["decrease_factor"])
        self.tolerance = float(state_dict["tolerance"])
        self.previous_metric = state_dict.get("previous_metric")
