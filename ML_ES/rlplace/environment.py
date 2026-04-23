"""RL environment and vectorized placement metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from placement import overlap_repulsion_loss, wirelength_attraction_loss

from .types import PlacementInstance
from .utils import radial_initialize_positions


@dataclass
class RewardConfig:
    """Reward shaping weights for policy search."""

    overlap_delta_weight: float
    wirelength_delta_weight: float
    overlap_ratio_delta_weight: float
    action_penalty_weight: float
    zero_overlap_bonus: float


def apply_positions(instance: PlacementInstance, positions: torch.Tensor) -> torch.Tensor:
    """Inject positions into a cloned cell feature tensor."""
    current = instance.cell_features.clone()
    current[:, 2:4] = positions
    return current


def overlap_ratio_from_cells(cell_features: torch.Tensor) -> float:
    """Compute exact overlap ratio using vectorized tensor operations."""
    num_cells = cell_features.shape[0]
    if num_cells <= 1:
        return 0.0

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]

    dx = torch.abs(positions[:, 0].unsqueeze(1) - positions[:, 0].unsqueeze(0))
    dy = torch.abs(positions[:, 1].unsqueeze(1) - positions[:, 1].unsqueeze(0))
    min_sep_x = (widths.unsqueeze(1) + widths.unsqueeze(0)) / 2.0
    min_sep_y = (heights.unsqueeze(1) + heights.unsqueeze(0)) / 2.0

    overlaps = (min_sep_x > dx) & (min_sep_y > dy)
    overlaps.fill_diagonal_(False)
    cells_with_overlap = overlaps.any(dim=1)
    return float(cells_with_overlap.float().mean().item())


def evaluate_positions(instance: PlacementInstance, positions: torch.Tensor) -> dict[str, float]:
    """Evaluate one placement state for reward shaping and validation."""
    with torch.no_grad():
        current = apply_positions(instance, positions)
        proxy_overlap = float(overlap_repulsion_loss(current, instance.pin_features, instance.edge_list).item())
        average_wl = float(wirelength_attraction_loss(current, instance.pin_features, instance.edge_list).item())
        normalized_wl = average_wl / math.sqrt(max(instance.total_area, 1.0))
        overlap_ratio = overlap_ratio_from_cells(current)
    return {
        "proxy_overlap": proxy_overlap,
        "normalized_wl": normalized_wl,
        "overlap_ratio": overlap_ratio,
    }


class PlacementEnv:
    """Simple continuous-action environment for one placement instance."""

    def __init__(
        self,
        instance: PlacementInstance,
        reward_config: RewardConfig,
        *,
        action_scale: float,
        max_steps: int,
        seed: int,
    ) -> None:
        self.instance = instance
        self.reward_config = reward_config
        self.action_scale = action_scale
        self.max_steps = max_steps
        self.seed = seed
        self.positions = torch.empty((instance.num_cells, 2), device=instance.cell_features.device)
        self.current_step = 0
        self.last_metrics: dict[str, float] | None = None

    def reset(self) -> tuple[torch.Tensor, dict[str, float]]:
        """Reset to the starter repo's radial initialization."""
        self.positions = radial_initialize_positions(self.instance.cell_features, self.seed)
        self.current_step = 0
        self.last_metrics = evaluate_positions(self.instance, self.positions)
        return self.positions.clone(), dict(self.last_metrics)

    def step_fraction(self) -> float:
        """Return normalized step progress in [0, 1]."""
        return self.current_step / max(self.max_steps - 1, 1)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, float]]:
        """Apply one bounded action to all cells."""
        bounded_action = torch.tanh(action.to(self.positions.device))
        scale = self.instance.cell_features[:, 4:6] * self.action_scale
        delta = bounded_action * scale
        self.positions = self.positions + delta
        self.current_step += 1

        metrics = evaluate_positions(self.instance, self.positions)
        assert self.last_metrics is not None

        reward = 0.0
        reward += self.reward_config.overlap_delta_weight * (
            self.last_metrics["proxy_overlap"] - metrics["proxy_overlap"]
        )
        reward += self.reward_config.overlap_ratio_delta_weight * (
            self.last_metrics["overlap_ratio"] - metrics["overlap_ratio"]
        )
        reward += self.reward_config.wirelength_delta_weight * (
            self.last_metrics["normalized_wl"] - metrics["normalized_wl"]
        )
        reward -= self.reward_config.action_penalty_weight * float(delta.square().mean().item())
        if metrics["overlap_ratio"] == 0.0:
            reward += self.reward_config.zero_overlap_bonus

        self.last_metrics = metrics
        done = metrics["overlap_ratio"] == 0.0 or self.current_step >= self.max_steps
        return self.positions.clone(), reward, done, metrics
