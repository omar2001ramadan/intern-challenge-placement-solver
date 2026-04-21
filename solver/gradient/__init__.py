"""Gradient-based global placement and soft wirelength refinement."""

import torch
import torch.optim as optim

from solver.core import (
    _calculate_normalized_metrics_fast,
    _choose_num_epochs,
    _get_candidate_pairs_kdtree,
    _use_candidate_pairs,
    overlap_repulsion_loss,
    wirelength_attraction_loss,
)
from solver.local_search import _refine_wirelength_with_bandit_projection


def _choose_refine_steps(num_cells):
    """Choose a modest post-legalization refinement budget.

    The main optimizer is good at reaching legality quickly, but it often stops
    with avoidable whitespace because overlap is the dominant objective. A short
    second pass from the best legal placement can usually trim wirelength
    without breaking legality.
    """
    if num_cells <= 40:
        return 180
    if num_cells <= 100:
        return 150
    if num_cells <= 300:
        return 100
    return 35


def _refine_wirelength_from_legal_placement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Refine wirelength starting from the best legal placement found so far.

    This is a soft constrained second phase: allow the optimizer to explore, but
    only keep snapshots that remain overlap-free. In practice this acts like a
    lightweight legalizer-aware fine-tuning stage and often removes a noticeable
    amount of wirelength the main phase leaves behind.
    """
    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    N = start_cell_features.shape[0]
    refine_steps = _choose_refine_steps(N)
    if refine_steps <= 0:
        return start_cell_features

    positions = start_cell_features[:, 2:4].clone().detach()
    positions.requires_grad_(True)

    optimizer = optim.Adam([positions], lr=0.05)
    best_cell_features = start_cell_features.clone()
    best_normalized_wl = start_metrics["normalized_wl"]

    use_candidate_pairs = _use_candidate_pairs(N)
    candidate_pairs = None
    refresh_pairs_every = 5
    eval_every = 5

    for step in range(refine_steps):
        if use_candidate_pairs and (step % refresh_pairs_every == 0 or candidate_pairs is None):
            current_features = base_cell_features.clone()
            current_features[:, 2:4] = positions.detach()
            candidate_pairs = _get_candidate_pairs_kdtree(current_features, extra_margin=1.0)

        optimizer.zero_grad()

        current_features = base_cell_features.clone()
        current_features[:, 2:4] = positions

        wl_loss = wirelength_attraction_loss(current_features, pin_features, edge_list)
        overlap_loss = overlap_repulsion_loss(
            current_features,
            pin_features,
            edge_list,
            pairs=candidate_pairs if use_candidate_pairs else None,
        )
        total_loss = wl_loss + 5.0 * overlap_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([positions], max_norm=1.0)
        optimizer.step()

        should_eval = (step % eval_every == 0) or (step == refine_steps - 1)
        if should_eval:
            eval_features = base_cell_features.clone()
            eval_features[:, 2:4] = positions.detach()
            metrics = _calculate_normalized_metrics_fast(eval_features, pin_features, edge_list)
            if metrics["overlap_ratio"] == 0.0 and metrics["normalized_wl"] < best_normalized_wl:
                best_normalized_wl = metrics["normalized_wl"]
                best_cell_features = eval_features.clone()
                if verbose:
                    print(
                        f"  Refinement step {step}/{refine_steps}: "
                        f"normalized_wl={best_normalized_wl:.6f}"
                    )

    return best_cell_features




def _single_train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.01,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    verbose=True,
    log_interval=100,
):
    """Train the placement optimization using momentum SGD.

    The baseline challenge uses a single static overlap loss on all pairs. This
    version makes two practical changes:
      1. A stronger overlap penalty that is much harder to "ignore" near zero.
      2. KD-tree candidate pruning on larger designs so the largest public test
         case remains fast.

    The public benchmark mostly cares about eliminating overlap, so model
    selection is done lexicographically: overlap first, wirelength second.
    """
    # Clone features and create learnable positions
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    N = cell_features.shape[0]
    num_epochs = _choose_num_epochs(N, num_epochs)
    if lr == 0.01:
        lr = 0.5
    if lambda_overlap == 10.0:
        lambda_overlap = 50.0

    # Make only cell positions require gradients
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    optimizer = optim.SGD([cell_positions], lr=lr, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
    }

    best_cell_features = None
    best_score = float("inf")
    zero_overlap_streak = 0
    eval_interval = max(50, num_epochs // 8)
    refresh_pairs_every = 20
    candidate_pairs = None
    use_candidate_pairs = _use_candidate_pairs(N)

    for epoch in range(num_epochs):
        if use_candidate_pairs and (epoch % refresh_pairs_every == 0 or candidate_pairs is None):
            current_features = cell_features.clone()
            current_features[:, 2:4] = cell_positions.detach()
            candidate_pairs = _get_candidate_pairs_kdtree(current_features)

        optimizer.zero_grad()

        current_features = cell_features.clone()
        current_features[:, 2:4] = cell_positions

        wl_loss = wirelength_attraction_loss(current_features, pin_features, edge_list)
        overlap_loss = overlap_repulsion_loss(
            current_features,
            pin_features,
            edge_list,
            pairs=candidate_pairs if use_candidate_pairs else None,
        )
        total_loss = lambda_wirelength * wl_loss + lambda_overlap * overlap_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())

        should_eval = (epoch % eval_interval == 0) or (epoch == num_epochs - 1)
        if should_eval:
            eval_features = cell_features.clone()
            eval_features[:, 2:4] = cell_positions.detach()
            metrics = _calculate_normalized_metrics_fast(eval_features, pin_features, edge_list)
            score = metrics["overlap_ratio"] * 1000.0 + metrics["normalized_wl"]

            if score < best_score:
                best_score = score
                best_cell_features = eval_features.clone()

            if metrics["overlap_ratio"] == 0.0:
                zero_overlap_streak += 1
            else:
                zero_overlap_streak = 0

            if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch}/{num_epochs}:")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Wirelength Loss: {wl_loss.item():.6f}")
                print(f"  Overlap Loss: {overlap_loss.item():.6f}")
                print(
                    f"  Eval -> overlap_ratio={metrics['overlap_ratio']:.6f}, "
                    f"normalized_wl={metrics['normalized_wl']:.6f}"
                )

            # Once overlap has been zero for several evaluation checkpoints, the
            # remaining improvements are usually tiny. Stop early.
            if zero_overlap_streak >= 3 and epoch >= int(0.6 * num_epochs):
                break

    if best_cell_features is None:
        best_cell_features = cell_features.clone()
        best_cell_features[:, 2:4] = cell_positions.detach()

    best_cell_features = _refine_wirelength_from_legal_placement(
        cell_features,
        best_cell_features,
        pin_features,
        edge_list,
        verbose=verbose,
    )
    best_cell_features = _refine_wirelength_with_bandit_projection(
        cell_features,
        best_cell_features,
        pin_features,
        edge_list,
        verbose=verbose,
    )

    return {
        "final_cell_features": best_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }


