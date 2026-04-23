"""Stage 2: distill a smaller student directly from the teacher policy."""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn.functional as F

from rlplace.environment import PlacementEnv, RewardConfig
from rlplace.evaluate import evaluate_policy_suite
from rlplace.graph import build_cell_graph, build_distillation_batch, build_single_graph_batch
from rlplace.models import PlacementPolicy, checkpoint_payload
from rlplace.training import JsonlLogger, LossAdaptiveLRScheduler, TerminationMonitor, atomic_torch_save
from rlplace.types import DistillSample
from rlplace.utils import (
    build_instance,
    choose_curriculum_case,
    load_json_config,
    move_instance_to_device,
    seed_everything,
    select_device,
)


def collect_teacher_samples(
    teacher: PlacementPolicy,
    config: dict,
    reward_config: RewardConfig,
    device: torch.device,
    rng: random.Random,
    epoch: int,
) -> list[DistillSample]:
    """Collect on-policy teacher supervision without a precomputed dataset."""
    samples: list[DistillSample] = []
    target_count = int(config["samples_per_epoch"])

    while len(samples) < target_count:
        case = choose_curriculum_case(config["curriculum"], epoch, int(config["epochs"]), rng)
        seed = int(config["seed"]) + epoch * 1000 + len(samples)
        instance = build_instance(case["num_macros"], case["num_std_cells"], seed, build_cell_graph)
        instance = move_instance_to_device(instance, device)
        env = PlacementEnv(
            instance,
            reward_config,
            action_scale=float(config["environment"]["action_scale"]),
            max_steps=int(config["environment"]["max_steps"]),
            seed=seed,
        )
        positions, _ = env.reset()
        done = False
        while not done and len(samples) < target_count:
            step_fraction = env.step_fraction()
            batch = build_single_graph_batch(instance, positions, step_fraction)
            outputs = teacher(batch)
            target_action = outputs["action_mean"].detach()
            samples.append(
                DistillSample(
                    instance=instance,
                    positions=positions.detach().clone(),
                    step_fraction=step_fraction,
                    target_action=target_action.clone(),
                )
            )
            positions, _, done, _ = env.step(target_action)

    return samples


def validation_distillation_loss(
    student: PlacementPolicy,
    teacher: PlacementPolicy,
    validation_cases: list[dict],
    reward_config: RewardConfig,
    action_scale: float,
    max_steps: int,
    device: torch.device,
    action_weight: float,
) -> float:
    """Compute held-out student-vs-teacher loss on fixed validation rollouts."""
    losses = []
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for case in validation_cases:
            instance = build_instance(case["num_macros"], case["num_std_cells"], case["seed"], build_cell_graph)
            instance = move_instance_to_device(instance, device)
            env = PlacementEnv(
                instance,
                reward_config,
                action_scale=action_scale,
                max_steps=max_steps,
                seed=case["seed"],
            )
            positions, _ = env.reset()
            done = False
            while not done:
                batch = build_single_graph_batch(instance, positions, env.step_fraction())
                teacher_outputs = teacher(batch)
                student_outputs = student(batch)
                action_loss = F.smooth_l1_loss(student_outputs["action_mean"], teacher_outputs["action_mean"])
                losses.append(float((action_weight * action_loss).item()))
                positions, _, done, _ = env.step(teacher_outputs["action_mean"].detach())
    return sum(losses) / max(len(losses), 1)


def default_validation_cases(curriculum: list[dict]) -> list[dict]:
    """Create a small deterministic validation suite from the curriculum."""
    return [
        {
            "num_macros": case["num_macros"],
            "num_std_cells": case["num_std_cells"],
            "seed": 9101 + idx,
        }
        for idx, case in enumerate(curriculum[: min(len(curriculum), 4)])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/student_distill.json")
    args = parser.parse_args()
    config = load_json_config(args.config)

    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(config.get("device", "auto"))
    rng = random.Random(seed)
    reward_config = RewardConfig(**config["environment"]["reward"])
    validation_cases = config.get("validation_cases") or default_validation_cases(config["curriculum"])

    teacher_checkpoint = torch.load(config["teacher_checkpoint"], map_location=device, weights_only=False)
    teacher = PlacementPolicy(**teacher_checkpoint["model_config"]).to(device)
    teacher.load_state_dict(teacher_checkpoint["state_dict"])
    teacher.eval()

    student = PlacementPolicy(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    scheduler = LossAdaptiveLRScheduler(optimizer, **config["optimizer"]["scheduler"])
    logger = JsonlLogger(config["log_path"])
    termination = TerminationMonitor()
    termination.install()

    start_epoch = 0
    last_completed_epoch = -1
    best_loss = float("inf")
    history = {
        "train_loss": [],
        "action_loss": [],
        "validation_loss": [],
        "validation_overlap": [],
        "validation_wl": [],
    }

    if config.get("resume") and os.path.exists(config["state_path"]):
        state = torch.load(config["state_path"], map_location=device, weights_only=False)
        student.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state.get("best_loss", best_loss))
        rng.setstate(state["python_random_state"])
        torch.random.set_rng_state(state["torch_random_state"])

    for epoch in range(start_epoch, int(config["epochs"])):
        if termination.should_stop:
            break

        teacher.eval()
        with torch.no_grad():
            samples = collect_teacher_samples(teacher, config, reward_config, device, rng, epoch)

        rng.shuffle(samples)
        total_loss = 0.0
        total_action_loss = 0.0
        num_batches = 0
        student.train()

        for start in range(0, len(samples), int(config["batch_size"])):
            batch_samples = samples[start : start + int(config["batch_size"])]
            batch = build_distillation_batch(batch_samples, device)
            outputs = student(batch)
            action_loss = F.smooth_l1_loss(outputs["action_mean"], batch.target_action)
            loss = float(config["loss_weights"]["action"]) * action_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(config["optimizer"]["grad_clip"]))
            optimizer.step()

            total_loss += float(loss.item())
            total_action_loss += float(action_loss.item())
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        validation_loss = validation_distillation_loss(
            student,
            teacher,
            validation_cases,
            reward_config,
            float(config["environment"]["action_scale"]),
            int(config["environment"]["max_steps"]),
            device,
            float(config["loss_weights"]["action"]),
        )
        validation = evaluate_policy_suite(
            student.eval(),
            validation_cases,
            device=device,
            reward_config=reward_config,
            action_scale=float(config["environment"]["action_scale"]),
            max_steps=int(config["environment"]["max_steps"]),
        )
        current_lr = scheduler.step(validation_loss)

        history["train_loss"].append(train_loss)
        history["action_loss"].append(total_action_loss / max(num_batches, 1))
        history["validation_loss"].append(validation_loss)
        history["validation_overlap"].append(validation["avg_overlap"])
        history["validation_wl"].append(validation["avg_normalized_wl"])

        logger.log(
            {
                "epoch": epoch,
                "train_loss": history["train_loss"][-1],
                "action_loss": history["action_loss"][-1],
                "validation_loss": validation_loss,
                "validation_overlap": validation["avg_overlap"],
                "validation_wl": validation["avg_normalized_wl"],
                "lr": current_lr,
            }
        )

        state = {
            "epoch": epoch,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_loss": best_loss,
            "config": config,
            "python_random_state": rng.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        atomic_torch_save(state, config["state_path"])

        if validation_loss < best_loss:
            best_loss = validation_loss
            atomic_torch_save(
                checkpoint_payload(
                    student,
                    config["model"],
                    extra={
                        "config": config,
                        "epoch": epoch,
                        "validation": validation,
                        "stage": "student_distill",
                    },
                ),
                config["best_path"],
            )
        last_completed_epoch = epoch

    atomic_torch_save(
        checkpoint_payload(
            student,
            config["model"],
            extra={
                "config": config,
                "epoch": last_completed_epoch,
                "stage": "student_distill",
            },
        ),
        config["final_path"],
    )


if __name__ == "__main__":
    main()
