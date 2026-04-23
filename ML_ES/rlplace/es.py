"""Evolution-strategy helpers for policy-only placement search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .environment import PlacementEnv, RewardConfig
from .evaluate import benchmark_cost
from .graph import build_single_graph_batch
from .models import PlacementPolicy
from .types import PlacementInstance


@dataclass
class EpisodeSpec:
    """One fixed training episode definition for ES evaluation."""

    instance: PlacementInstance
    seed: int


@dataclass
class ESConfig:
    """Hyperparameters for one mirrored evolution-strategy update."""

    sigma: float
    perturbation_pairs: int
    grad_clip: float


def evaluate_episode_metrics(
    policy: PlacementPolicy,
    episode: EpisodeSpec,
    *,
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
) -> dict[str, float]:
    """Run one deterministic episode and return final benchmark metrics."""
    env = PlacementEnv(
        instance=episode.instance,
        reward_config=reward_config,
        action_scale=action_scale,
        max_steps=max_steps,
        seed=episode.seed,
    )
    positions, _ = env.reset()
    done = False
    final_metrics = env.last_metrics or {}

    while not done:
        batch = build_single_graph_batch(env.instance, positions, env.step_fraction())
        with torch.no_grad():
            action = policy(batch)["action_mean"]
        positions, _, done, final_metrics = env.step(action.detach())

    return {
        "overlap_ratio": float(final_metrics["overlap_ratio"]),
        "normalized_wl": float(final_metrics["normalized_wl"]),
    }


def evaluate_episode_batch(
    policy: PlacementPolicy,
    episodes: Sequence[EpisodeSpec],
    *,
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
) -> dict[str, float]:
    """Average deterministic final metrics and benchmark score over one batch."""
    overlaps = []
    wirelengths = []

    for episode in episodes:
        metrics = evaluate_episode_metrics(
            policy,
            episode,
            reward_config=reward_config,
            action_scale=action_scale,
            max_steps=max_steps,
        )
        overlaps.append(metrics["overlap_ratio"])
        wirelengths.append(metrics["normalized_wl"])

    avg_overlap = sum(overlaps) / max(len(overlaps), 1)
    avg_wl = sum(wirelengths) / max(len(wirelengths), 1)
    cost = benchmark_cost(avg_overlap, avg_wl)
    return {
        "avg_overlap": avg_overlap,
        "avg_normalized_wl": avg_wl,
        "benchmark_cost": cost,
        "benchmark_score": -cost,
    }


def _trainable_parameters(policy: PlacementPolicy) -> list[torch.nn.Parameter]:
    return [param for param in policy.parameters() if param.requires_grad]


def _sample_noises(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.randn_like(param) for param in parameters]


def _apply_noise(parameters: Sequence[torch.nn.Parameter], noises: Sequence[torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for param, noise in zip(parameters, noises):
            param.add_(noise, alpha=scale)


def es_gradient_step(
    policy: PlacementPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[EpisodeSpec],
    *,
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
    config: ESConfig,
) -> dict[str, float]:
    """Estimate a benchmark-score gradient with mirrored perturbations and apply one optimizer step."""
    parameters = _trainable_parameters(policy)
    grad_buffers = [torch.zeros_like(param) for param in parameters]
    fitness_centers = []
    fitness_deltas = []
    overlap_centers = []
    wl_centers = []

    for _ in range(config.perturbation_pairs):
        noises = _sample_noises(parameters)

        _apply_noise(parameters, noises, config.sigma)
        plus_metrics = evaluate_episode_batch(
            policy,
            episodes,
            reward_config=reward_config,
            action_scale=action_scale,
            max_steps=max_steps,
        )

        _apply_noise(parameters, noises, -2.0 * config.sigma)
        minus_metrics = evaluate_episode_batch(
            policy,
            episodes,
            reward_config=reward_config,
            action_scale=action_scale,
            max_steps=max_steps,
        )

        _apply_noise(parameters, noises, config.sigma)

        fitness_delta = plus_metrics["benchmark_score"] - minus_metrics["benchmark_score"]
        fitness_center = 0.5 * (plus_metrics["benchmark_score"] + minus_metrics["benchmark_score"])
        overlap_center = 0.5 * (plus_metrics["avg_overlap"] + minus_metrics["avg_overlap"])
        wl_center = 0.5 * (plus_metrics["avg_normalized_wl"] + minus_metrics["avg_normalized_wl"])

        fitness_deltas.append(fitness_delta)
        fitness_centers.append(fitness_center)
        overlap_centers.append(overlap_center)
        wl_centers.append(wl_center)

        for grad_buffer, noise in zip(grad_buffers, noises):
            grad_buffer.add_(noise, alpha=fitness_delta)

    scale = 1.0 / (2.0 * config.sigma * max(config.perturbation_pairs, 1))
    optimizer.zero_grad(set_to_none=True)
    for param, grad_buffer in zip(parameters, grad_buffers):
        param.grad = -grad_buffer.mul(scale)

    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm=config.grad_clip).item())
    optimizer.step()

    fitness_delta_mean = sum(fitness_deltas) / max(len(fitness_deltas), 1)
    fitness_delta_std = (
        sum((delta - fitness_delta_mean) ** 2 for delta in fitness_deltas) / max(len(fitness_deltas), 1)
    ) ** 0.5

    return {
        "population_fitness": sum(fitness_centers) / max(len(fitness_centers), 1),
        "population_overlap": sum(overlap_centers) / max(len(overlap_centers), 1),
        "population_wl": sum(wl_centers) / max(len(wl_centers), 1),
        "fitness_delta_mean": fitness_delta_mean,
        "fitness_delta_std": fitness_delta_std,
        "gradient_norm": gradient_norm,
    }
