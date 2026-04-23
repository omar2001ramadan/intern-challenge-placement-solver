"""Shared dataclasses for RL placement training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlacementInstance:
    """One generated placement instance plus its derived cell graph."""

    cell_features: torch.Tensor
    pin_features: torch.Tensor
    edge_list: torch.Tensor
    cell_edge_index: torch.Tensor
    cell_edge_weight: torch.Tensor
    total_area: float
    num_cells: int


@dataclass
class GraphBatch:
    """Batch of variable-sized placement graphs."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    batch_index: torch.Tensor
    sizes: torch.Tensor
    target_action: torch.Tensor | None = None


@dataclass
class DistillSample:
    """One teacher state-action supervision sample."""

    instance: PlacementInstance
    positions: torch.Tensor
    step_fraction: float
    target_action: torch.Tensor
