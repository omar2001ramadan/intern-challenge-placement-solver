"""General helpers for RL placement training."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

import torch

from placement import generate_placement_input

from .types import PlacementInstance


def load_json_config(path: str | Path) -> dict:
    """Load a JSON config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_device(device_name: str) -> torch.device:
    """Resolve a requested device name."""
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def seed_everything(seed: int) -> None:
    """Seed Python and torch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)


def graph_mean_pool(node_embeddings: torch.Tensor, batch_index: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Mean-pool node embeddings into graph embeddings."""
    pooled = torch.zeros((batch_size, node_embeddings.shape[1]), device=node_embeddings.device)
    pooled.index_add_(0, batch_index, node_embeddings)
    counts = torch.zeros((batch_size, 1), device=node_embeddings.device)
    counts.index_add_(0, batch_index, torch.ones((batch_index.shape[0], 1), device=node_embeddings.device))
    return pooled / counts.clamp(min=1.0)


def radial_initialize_positions(cell_features: torch.Tensor, seed: int) -> torch.Tensor:
    """Match the starter repo's radial random initialization."""
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = math.sqrt(total_area) * 0.6
    angles = torch.rand(total_cells, generator=cpu_generator) * 2 * math.pi
    radii = torch.rand(total_cells, generator=cpu_generator) * spread_radius
    positions = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
    return positions.to(cell_features.device)


def choose_curriculum_case(curriculum: Sequence[dict], epoch: int, total_epochs: int, rng: random.Random) -> dict:
    """Expose larger cases gradually over training."""
    usable = max(1, math.ceil(((epoch + 1) / max(total_epochs, 1)) * len(curriculum)))
    return dict(rng.choice(list(curriculum[:usable])))


def move_instance_to_device(instance: PlacementInstance, device: torch.device) -> PlacementInstance:
    """Move one placement instance to the requested device."""
    return PlacementInstance(
        cell_features=instance.cell_features.to(device),
        pin_features=instance.pin_features.to(device),
        edge_list=instance.edge_list.to(device),
        cell_edge_index=instance.cell_edge_index.to(device),
        cell_edge_weight=instance.cell_edge_weight.to(device),
        total_area=instance.total_area,
        num_cells=instance.num_cells,
    )


def build_instance(num_macros: int, num_std_cells: int, seed: int, graph_builder) -> PlacementInstance:
    """Generate one placement instance and derive its cell graph."""
    previous_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    finally:
        torch.random.set_rng_state(previous_state)
    cell_edge_index, cell_edge_weight = graph_builder(pin_features, edge_list, cell_features.shape[0])
    return PlacementInstance(
        cell_features=cell_features,
        pin_features=pin_features,
        edge_list=edge_list,
        cell_edge_index=cell_edge_index,
        cell_edge_weight=cell_edge_weight,
        total_area=float(cell_features[:, 0].sum().item()),
        num_cells=int(cell_features.shape[0]),
    )
