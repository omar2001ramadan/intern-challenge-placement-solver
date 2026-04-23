"""Evaluate a trained teacher or student policy on the starter test cases."""

from __future__ import annotations

import argparse

import torch

from rlplace.environment import RewardConfig
from rlplace.evaluate import run_policy_episode
from rlplace.graph import build_cell_graph
from rlplace.models import PlacementPolicy
from rlplace.utils import build_instance, load_json_config, move_instance_to_device, select_device
from test import TEST_CASES


def parse_test_range(spec: str) -> tuple[int, int]:
    """Parse 1-based inclusive test ranges like 1:10."""
    start, end = spec.split(":", maxsplit=1)
    return int(start), int(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_policy.json")
    parser.add_argument("--tests", default="1:10")
    args = parser.parse_args()
    config = load_json_config(args.config)

    device = select_device(config.get("device", "auto"))
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    policy = PlacementPolicy(**checkpoint["model_config"]).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    reward_config = RewardConfig(**config["environment"]["reward"])
    start, end = parse_test_range(args.tests)
    selected_cases = TEST_CASES[start - 1 : end]

    overlap_values = []
    wl_values = []

    for test_id, num_macros, num_std_cells, seed in selected_cases:
        instance = build_instance(num_macros, num_std_cells, seed, build_cell_graph)
        instance = move_instance_to_device(instance, device)
        metrics = run_policy_episode(
            policy,
            instance,
            reward_config=reward_config,
            action_scale=float(config["environment"]["action_scale"]),
            max_steps=int(config["environment"]["max_steps"]),
            seed=seed,
        )
        overlap_values.append(metrics["overlap_ratio"])
        wl_values.append(metrics["normalized_wl"])
        print(
            f"Test {test_id}: overlap={metrics['overlap_ratio']:.4f}, "
            f"normalized_wl={metrics['normalized_wl']:.4f}"
        )

    print(f"Average Overlap: {sum(overlap_values) / max(len(overlap_values), 1):.4f}")
    print(f"Average Wirelength: {sum(wl_values) / max(len(wl_values), 1):.4f}")


if __name__ == "__main__":
    main()
