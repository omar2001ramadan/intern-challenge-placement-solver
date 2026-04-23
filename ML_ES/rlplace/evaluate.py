"""Deterministic policy evaluation helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from placement import calculate_normalized_metrics

from .environment import PlacementEnv, RewardConfig, apply_positions
from .graph import build_cell_graph, build_single_graph_batch
from .models import PlacementPolicy
from .utils import build_instance, move_instance_to_device

BENCHMARK_OVERLAP_SCALE = 1_000_000.0


def benchmark_cost(avg_overlap: float, avg_normalized_wl: float) -> float:
    """Scalarize the lexicographic benchmark: overlap first, wirelength second."""
    return BENCHMARK_OVERLAP_SCALE * avg_overlap + avg_normalized_wl


def run_policy_episode(
    policy: PlacementPolicy,
    instance,
    *,
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
    seed: int,
) -> dict[str, float]:
    """Run one deterministic rollout and report starter-suite metrics."""
    env = PlacementEnv(
        instance=instance,
        reward_config=reward_config,
        action_scale=action_scale,
        max_steps=max_steps,
        seed=seed,
    )
    positions, _ = env.reset()
    done = False
    while not done:
        batch = build_single_graph_batch(env.instance, positions, env.step_fraction())
        with torch.no_grad():
            action = policy(batch)["action_mean"]
        positions, _, done, _ = env.step(action.detach())

    final_cell_features = apply_positions(instance, positions).detach().cpu()
    metrics = calculate_normalized_metrics(
        final_cell_features,
        instance.pin_features.detach().cpu(),
        instance.edge_list.detach().cpu(),
    )
    return {
        "overlap_ratio": float(metrics["overlap_ratio"]),
        "normalized_wl": float(metrics["normalized_wl"]),
    }


def evaluate_policy_suite(
    policy: PlacementPolicy,
    validation_cases: Sequence[dict],
    *,
    device,
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
) -> dict[str, float]:
    """Run deterministic evaluation over a small fixed suite."""
    overlap_values = []
    wl_values = []
    for case in validation_cases:
        instance = build_instance(
            case["num_macros"],
            case["num_std_cells"],
            case["seed"],
            build_cell_graph,
        )
        instance = move_instance_to_device(instance, device)
        metrics = run_policy_episode(
            policy,
            instance,
            reward_config=reward_config,
            action_scale=action_scale,
            max_steps=max_steps,
            seed=case["seed"],
        )
        overlap_values.append(metrics["overlap_ratio"])
        wl_values.append(metrics["normalized_wl"])

    avg_overlap = sum(overlap_values) / max(len(overlap_values), 1)
    avg_wl = sum(wl_values) / max(len(wl_values), 1)
    cost = benchmark_cost(avg_overlap, avg_wl)
    return {
        "avg_overlap": avg_overlap,
        "avg_normalized_wl": avg_wl,
        "benchmark_cost": cost,
        "benchmark_score": -cost,
    }
