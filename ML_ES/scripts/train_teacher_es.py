"""Stage 1: train a large teacher with evolution strategies from scratch."""

from __future__ import annotations

import argparse
import os
import random

import torch

from rlplace.environment import RewardConfig
from rlplace.es import ESConfig, EpisodeSpec, es_gradient_step
from rlplace.evaluate import evaluate_policy_suite
from rlplace.graph import build_cell_graph
from rlplace.models import PlacementPolicy, checkpoint_payload
from rlplace.training import JsonlLogger, LossAdaptiveLRScheduler, TerminationMonitor, atomic_torch_save
from rlplace.utils import (
    build_instance,
    choose_curriculum_case,
    load_json_config,
    move_instance_to_device,
    seed_everything,
    select_device,
)


def default_validation_cases(curriculum: list[dict]) -> list[dict]:
    """Create a small deterministic validation suite from the curriculum."""
    return [
        {
            "num_macros": case["num_macros"],
            "num_std_cells": case["num_std_cells"],
            "seed": 9001 + idx,
        }
        for idx, case in enumerate(curriculum[: min(len(curriculum), 4)])
    ]
def build_epoch_episodes(
    config: dict,
    *,
    epoch: int,
    rng: random.Random,
    seed: int,
    device: torch.device,
) -> list[EpisodeSpec]:
    """Build a fixed batch of environments for one ES generation."""
    episodes = []
    for environment_idx in range(int(config["environments_per_epoch"])):
        case = choose_curriculum_case(config["curriculum"], epoch, int(config["epochs"]), rng)
        instance_seed = seed + epoch * 1000 + environment_idx
        instance = build_instance(
            case["num_macros"],
            case["num_std_cells"],
            instance_seed,
            build_cell_graph,
        )
        instance = move_instance_to_device(instance, device)
        episodes.append(EpisodeSpec(instance=instance, seed=instance_seed))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher_es.json")
    args = parser.parse_args()
    config = load_json_config(args.config)

    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(config.get("device", "auto"))
    rng = random.Random(seed)
    base_reward_config = RewardConfig(**config["environment"]["reward"])
    es_config = ESConfig(
        sigma=float(config["evolution"]["sigma"]),
        perturbation_pairs=int(config["evolution"]["perturbation_pairs"]),
        grad_clip=float(config["optimizer"]["grad_clip"]),
    )
    validation_cases = config.get("validation_cases") or default_validation_cases(config["curriculum"])

    policy = PlacementPolicy(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    scheduler = LossAdaptiveLRScheduler(optimizer, **config["optimizer"]["scheduler"])
    logger = JsonlLogger(config["log_path"])
    termination = TerminationMonitor()
    termination.install()

    start_epoch = 0
    last_completed_epoch = -1
    best_benchmark_cost = float("inf")
    history = {
        "mean_fitness": [],
        "population_overlap": [],
        "population_wl": [],
        "fitness_delta_mean": [],
        "fitness_delta_std": [],
        "gradient_norm": [],
        "validation_overlap": [],
        "validation_wl": [],
        "validation_benchmark_cost": [],
    }

    if config.get("resume") and os.path.exists(config["state_path"]):
        state = torch.load(config["state_path"], map_location=device, weights_only=False)
        policy.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
        best_benchmark_cost = float(state.get("best_benchmark_cost", best_benchmark_cost))
        rng.setstate(state["python_random_state"])
        torch.random.set_rng_state(state["torch_random_state"])

    for epoch in range(start_epoch, int(config["epochs"])):
        if termination.should_stop:
            break

        reward_config = base_reward_config
        episodes = build_epoch_episodes(config, epoch=epoch, rng=rng, seed=seed, device=device)

        policy.train()
        stats = es_gradient_step(
            policy,
            optimizer,
            episodes,
            reward_config=reward_config,
            action_scale=float(config["environment"]["action_scale"]),
            max_steps=int(config["environment"]["max_steps"]),
            config=es_config,
        )

        policy.eval()
        validation = evaluate_policy_suite(
            policy,
            validation_cases,
            device=device,
            reward_config=reward_config,
            action_scale=float(config["environment"]["action_scale"]),
            max_steps=int(config["environment"]["max_steps"]),
        )
        current_lr = scheduler.step(validation["benchmark_cost"])

        history["mean_fitness"].append(stats["population_fitness"])
        history["population_overlap"].append(stats["population_overlap"])
        history["population_wl"].append(stats["population_wl"])
        history["fitness_delta_mean"].append(stats["fitness_delta_mean"])
        history["fitness_delta_std"].append(stats["fitness_delta_std"])
        history["gradient_norm"].append(stats["gradient_norm"])
        history["validation_overlap"].append(validation["avg_overlap"])
        history["validation_wl"].append(validation["avg_normalized_wl"])
        history["validation_benchmark_cost"].append(validation["benchmark_cost"])

        logger.log(
            {
                "epoch": epoch,
                "mean_fitness": stats["population_fitness"],
                "population_overlap": stats["population_overlap"],
                "population_wl": stats["population_wl"],
                "fitness_delta_mean": stats["fitness_delta_mean"],
                "fitness_delta_std": stats["fitness_delta_std"],
                "gradient_norm": stats["gradient_norm"],
                "validation_overlap": validation["avg_overlap"],
                "validation_wl": validation["avg_normalized_wl"],
                "validation_benchmark_cost": validation["benchmark_cost"],
                "lr": current_lr,
            }
        )

        state = {
            "epoch": epoch,
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_benchmark_cost": best_benchmark_cost,
            "config": config,
            "python_random_state": rng.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        atomic_torch_save(state, config["state_path"])

        if validation["benchmark_cost"] < best_benchmark_cost:
            best_benchmark_cost = validation["benchmark_cost"]
            atomic_torch_save(
                checkpoint_payload(
                    policy,
                    config["model"],
                    extra={
                        "config": config,
                        "epoch": epoch,
                        "validation": validation,
                        "stage": "teacher_es",
                    },
                ),
                config["best_path"],
            )
        last_completed_epoch = epoch

    atomic_torch_save(
        checkpoint_payload(
            policy,
            config["model"],
            extra={
                "config": config,
                "epoch": last_completed_epoch,
                "stage": "teacher_es",
            },
        ),
        config["final_path"],
    )


if __name__ == "__main__":
    main()
