"""Cell-graph construction and graph feature collation."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

from .types import DistillSample, GraphBatch, PlacementInstance


def build_cell_graph(pin_features: torch.Tensor, edge_list: torch.Tensor, num_cells: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse pin-level connectivity into a weighted directed cell graph."""
    if edge_list.numel() == 0:
        edge_index = torch.arange(num_cells, dtype=torch.long).repeat(2, 1)
        edge_weight = torch.ones(num_cells, dtype=torch.float32)
        return edge_index, edge_weight

    pin_to_cell = pin_features[:, 0].long()
    weights: dict[tuple[int, int], float] = defaultdict(float)

    for src_pin, dst_pin in edge_list.long().tolist():
        src_cell = int(pin_to_cell[src_pin].item())
        dst_cell = int(pin_to_cell[dst_pin].item())
        if src_cell == dst_cell:
            continue
        pair = (min(src_cell, dst_cell), max(src_cell, dst_cell))
        weights[pair] += 1.0

    for idx in range(num_cells):
        weights[(idx, idx)] += 1.0

    directed_edges = []
    directed_weights = []
    for (src, dst), weight in weights.items():
        directed_edges.append((src, dst))
        directed_weights.append(weight)
        if src != dst:
            directed_edges.append((dst, src))
            directed_weights.append(weight)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(directed_weights, dtype=torch.float32)
    return edge_index, edge_weight


def node_features_from_state(instance: PlacementInstance, positions: torch.Tensor, step_fraction: float) -> torch.Tensor:
    """Build per-cell features for policy inference."""
    cell_features = instance.cell_features
    areas = cell_features[:, 0]
    num_pins = cell_features[:, 1]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    total_area = torch.clamp(areas.sum(), min=1.0)
    norm_scale = total_area.sqrt()
    centroid = positions.mean(dim=0, keepdim=True)
    rel_positions = positions - centroid
    centroid_distance = rel_positions.norm(dim=1, keepdim=True)
    macro_flag = (heights > 1.0).float().unsqueeze(1)

    degree = torch.zeros(instance.num_cells, 1, device=positions.device)
    degree.index_add_(
        0,
        instance.cell_edge_index[0],
        torch.ones((instance.cell_edge_index.shape[1], 1), device=positions.device),
    )

    return torch.cat(
        [
            (areas / total_area).unsqueeze(1),
            (widths / norm_scale).unsqueeze(1),
            (heights / norm_scale).unsqueeze(1),
            (num_pins / num_pins.clamp(min=1.0).max()).unsqueeze(1),
            degree / degree.clamp(min=1.0).max(),
            (positions[:, 0] / norm_scale).unsqueeze(1),
            (positions[:, 1] / norm_scale).unsqueeze(1),
            (rel_positions[:, 0] / norm_scale).unsqueeze(1),
            (rel_positions[:, 1] / norm_scale).unsqueeze(1),
            centroid_distance / norm_scale,
            macro_flag,
            torch.full((instance.num_cells, 1), float(step_fraction), device=positions.device),
        ],
        dim=1,
    )


def build_single_graph_batch(
    instance: PlacementInstance,
    positions: torch.Tensor,
    step_fraction: float,
) -> GraphBatch:
    """Wrap one placement state into the shared graph batch format."""
    return GraphBatch(
        node_features=node_features_from_state(instance, positions, step_fraction),
        edge_index=instance.cell_edge_index,
        edge_weight=instance.cell_edge_weight,
        batch_index=torch.zeros(instance.num_cells, dtype=torch.long, device=positions.device),
        sizes=torch.tensor([instance.num_cells], dtype=torch.long, device=positions.device),
    )


def build_distillation_batch(samples: Sequence[DistillSample], device: torch.device) -> GraphBatch:
    """Concatenate variable-sized distillation samples into one graph batch."""
    node_features = []
    edge_indices = []
    edge_weights = []
    batch_index = []
    sizes = []
    target_actions = []
    offset = 0

    for batch_id, sample in enumerate(samples):
        node_features.append(node_features_from_state(sample.instance, sample.positions, sample.step_fraction))
        edge_indices.append(sample.instance.cell_edge_index + offset)
        edge_weights.append(sample.instance.cell_edge_weight)
        batch_index.append(torch.full((sample.instance.num_cells,), batch_id, dtype=torch.long, device=device))
        sizes.append(sample.instance.num_cells)
        target_actions.append(sample.target_action)
        offset += sample.instance.num_cells

    return GraphBatch(
        node_features=torch.cat(node_features, dim=0).to(device),
        edge_index=torch.cat(edge_indices, dim=1).to(device),
        edge_weight=torch.cat(edge_weights, dim=0).to(device),
        batch_index=torch.cat(batch_index, dim=0),
        sizes=torch.tensor(sizes, dtype=torch.long, device=device),
        target_action=torch.cat(target_actions, dim=0).to(device),
    )
