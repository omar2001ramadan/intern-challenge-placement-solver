"""Legal detailed-placement moves that preserve zero overlap."""

import torch

from solver.core import (
    CellFeatureIdx,
    PinFeatureIdx,
    linear_sum_assignment,
    np,
    _calculate_normalized_metrics_fast,
)


def _build_incident_edge_lists(pin_features, edge_list, num_cells):
    """Build per-cell incident edge lists for exact local wirelength deltas."""
    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    incident_own = [[] for _ in range(num_cells)]
    incident_other = [[] for _ in range(num_cells)]

    for src_pin, tgt_pin in edge_list.detach().cpu().numpy():
        src_pin = int(src_pin)
        tgt_pin = int(tgt_pin)
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])

        incident_own[src_cell].append(src_pin)
        incident_other[src_cell].append(tgt_pin)
        if tgt_cell != src_cell:
            incident_own[tgt_cell].append(tgt_pin)
            incident_other[tgt_cell].append(src_pin)

    incident = []
    for own, other in zip(incident_own, incident_other):
        incident.append(
            (
                np.asarray(own, dtype=np.int32),
                np.asarray(other, dtype=np.int32),
            )
        )
    return pin_cell, incident


def _smooth_pair_cost_np(dx, dy, alpha=0.1):
    """Stable NumPy implementation of the smooth max used for wirelength."""
    m = np.maximum(dx, dy)
    return m + alpha * np.log(np.exp((dx - m) / alpha) + np.exp((dy - m) / alpha))


def _nearest_legal_x_np(cell_idx, target_x, fixed_y, positions, widths, heights, margin=1e-6):
    """Project a cell's x coordinate to the nearest legal value at fixed y."""
    cell_height = heights[cell_idx]
    cell_width = widths[cell_idx]

    blocker_mask = np.abs(positions[:, 1] - fixed_y) < (cell_height + heights) * 0.5 - 1e-12
    blocker_mask[cell_idx] = False
    blocker_idx = np.nonzero(blocker_mask)[0]
    if blocker_idx.size == 0:
        return float(target_x)

    sep_x = (cell_width + widths[blocker_idx]) * 0.5
    starts = positions[blocker_idx, 0] - sep_x - margin
    ends = positions[blocker_idx, 0] + sep_x + margin
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    left = starts[0]
    right = ends[0]
    containing_interval = None

    for start, end in zip(starts[1:], ends[1:]):
        if start <= right:
            right = max(right, end)
        else:
            if left <= target_x <= right:
                containing_interval = (left, right)
                break
            left, right = start, end

    if containing_interval is None and left <= target_x <= right:
        containing_interval = (left, right)

    if containing_interval is None:
        return float(target_x)

    left, right = containing_interval
    left_candidate = left - margin
    right_candidate = right + margin
    if abs(target_x - left_candidate) <= abs(right_candidate - target_x):
        return float(left_candidate)
    return float(right_candidate)


def _nearest_legal_y_np(cell_idx, target_y, fixed_x, positions, widths, heights, margin=1e-6):
    """Project a cell's y coordinate to the nearest legal value at fixed x."""
    cell_height = heights[cell_idx]
    cell_width = widths[cell_idx]

    blocker_mask = np.abs(positions[:, 0] - fixed_x) < (cell_width + widths) * 0.5 - 1e-12
    blocker_mask[cell_idx] = False
    blocker_idx = np.nonzero(blocker_mask)[0]
    if blocker_idx.size == 0:
        return float(target_y)

    sep_y = (cell_height + heights[blocker_idx]) * 0.5
    starts = positions[blocker_idx, 1] - sep_y - margin
    ends = positions[blocker_idx, 1] + sep_y + margin
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    left = starts[0]
    right = ends[0]
    containing_interval = None

    for start, end in zip(starts[1:], ends[1:]):
        if start <= right:
            right = max(right, end)
        else:
            if left <= target_y <= right:
                containing_interval = (left, right)
                break
            left, right = start, end

    if containing_interval is None and left <= target_y <= right:
        containing_interval = (left, right)

    if containing_interval is None:
        return float(target_y)

    left, right = containing_interval
    left_candidate = left - margin
    right_candidate = right + margin
    if abs(target_y - left_candidate) <= abs(right_candidate - target_y):
        return float(left_candidate)
    return float(right_candidate)


def _projected_target_local_search(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    target_modes=("median",),
    max_passes=4,
    max_cells_per_pass=None,
    verbose=False,
):
    """Legal local search with exact wirelength deltas.

    Each move computes a coordinate target from incident nets, then projects the
    moved cell onto the nearest non-overlapping x/y coordinate. A small UCB
    bandit orders the move operators so successful actions get tried first. The
    final move is still accepted only if it strictly improves the exact local
    wirelength contribution for that cell, so the search remains conservative.
    """
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    if num_cells <= 1:
        return start_cell_features

    pin_cell, incident = _build_incident_edge_lists(pin_features, edge_list, num_cells)
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy()
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy()

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64).copy()
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64).copy()
    areas = start_cell_features[:, CellFeatureIdx.AREA].detach().cpu().numpy().astype(np.float64).copy()

    if max_cells_per_pass is None or max_cells_per_pass > num_cells:
        max_cells_per_pass = num_cells

    degrees = np.asarray([len(own) for own, _ in incident], dtype=np.int32)
    priority = degrees + 0.05 * np.sqrt(areas)
    cell_order = np.argsort(-priority)[:max_cells_per_pass]

    op_specs = []
    for mode in target_modes:
        op_specs.extend(
            [
                ("x", mode),
                ("y", mode),
                ("xy", mode),
                ("yx", mode),
            ]
        )

    bandit_counts = np.ones(len(op_specs), dtype=np.float64)
    bandit_rewards = np.zeros(len(op_specs), dtype=np.float64)

    def local_wirelength(cell_idx, cand_x=None, cand_y=None):
        own_pins, other_pins = incident[cell_idx]
        if own_pins.size == 0:
            return 0.0

        x = positions[cell_idx, 0] if cand_x is None else cand_x
        y = positions[cell_idx, 1] if cand_y is None else cand_y

        other_abs_x = positions[pin_cell[other_pins], 0] + pin_x[other_pins]
        other_abs_y = positions[pin_cell[other_pins], 1] + pin_y[other_pins]
        own_abs_x = x + pin_x[own_pins]
        own_abs_y = y + pin_y[own_pins]

        dx = np.abs(own_abs_x - other_abs_x)
        dy = np.abs(own_abs_y - other_abs_y)
        return float(_smooth_pair_cost_np(dx, dy).sum())

    def target_positions(cell_idx):
        own_pins, other_pins = incident[cell_idx]
        if own_pins.size == 0:
            current_x = float(positions[cell_idx, 0])
            current_y = float(positions[cell_idx, 1])
            return {mode: (current_x, current_y) for mode in target_modes}

        target_x_values = positions[pin_cell[other_pins], 0] + pin_x[other_pins] - pin_x[own_pins]
        target_y_values = positions[pin_cell[other_pins], 1] + pin_y[other_pins] - pin_y[own_pins]

        targets = {}
        if "median" in target_modes:
            targets["median"] = (
                float(np.median(target_x_values)),
                float(np.median(target_y_values)),
            )
        if "mean" in target_modes:
            targets["mean"] = (
                float(target_x_values.mean()),
                float(target_y_values.mean()),
            )
        return targets

    best_cell_features = start_cell_features.clone()
    best_normalized_wl = start_metrics["normalized_wl"]

    for search_pass in range(max_passes):
        total_reward = 0.0
        moved_cells = 0

        for cell_idx in cell_order:
            own_pins, _ = incident[int(cell_idx)]
            if own_pins.size == 0:
                continue

            base_x = float(positions[cell_idx, 0])
            base_y = float(positions[cell_idx, 1])
            base_local_wl = local_wirelength(int(cell_idx), base_x, base_y)
            targets = target_positions(int(cell_idx))

            total_trials = bandit_counts.sum()
            ucb_scores = (bandit_rewards / bandit_counts) + 0.25 * np.sqrt(
                np.log(total_trials + 1.0) / bandit_counts
            )
            op_order = np.argsort(-ucb_scores)

            best_local = (base_local_wl, base_x, base_y, None)

            for op_idx in op_order:
                move_axis, target_mode = op_specs[int(op_idx)]
                target_x, target_y = targets[target_mode]

                if move_axis == "x":
                    cand_x = _nearest_legal_x_np(int(cell_idx), target_x, base_y, positions, widths, heights)
                    cand_y = base_y
                elif move_axis == "y":
                    cand_x = base_x
                    cand_y = _nearest_legal_y_np(int(cell_idx), target_y, base_x, positions, widths, heights)
                elif move_axis == "xy":
                    cand_x = _nearest_legal_x_np(int(cell_idx), target_x, base_y, positions, widths, heights)
                    cand_y = _nearest_legal_y_np(int(cell_idx), target_y, cand_x, positions, widths, heights)
                else:
                    cand_y = _nearest_legal_y_np(int(cell_idx), target_y, base_x, positions, widths, heights)
                    cand_x = _nearest_legal_x_np(int(cell_idx), target_x, cand_y, positions, widths, heights)

                if abs(cand_x - base_x) < 1e-12 and abs(cand_y - base_y) < 1e-12:
                    continue

                cand_local_wl = local_wirelength(int(cell_idx), cand_x, cand_y)
                if cand_local_wl + 1e-9 < best_local[0]:
                    best_local = (cand_local_wl, cand_x, cand_y, int(op_idx))

            if best_local[3] is not None:
                reward = base_local_wl - best_local[0]
                positions[cell_idx, 0] = best_local[1]
                positions[cell_idx, 1] = best_local[2]
                bandit_counts[best_local[3]] += 1.0
                bandit_rewards[best_local[3]] += reward
                total_reward += reward
                moved_cells += 1

        candidate_features = base_cell_features.clone()
        candidate_features[:, 2] = torch.from_numpy(positions[:, 0]).to(candidate_features.dtype)
        candidate_features[:, 3] = torch.from_numpy(positions[:, 1]).to(candidate_features.dtype)
        metrics = _calculate_normalized_metrics_fast(candidate_features, pin_features, edge_list)

        if metrics["overlap_ratio"] == 0.0 and metrics["normalized_wl"] < best_normalized_wl:
            best_normalized_wl = metrics["normalized_wl"]
            best_cell_features = candidate_features.clone()

        if verbose:
            print(
                f"  Bandit pass {search_pass}/{max_passes}: "
                f"moved={moved_cells}, reward={total_reward:.3f}, "
                f"normalized_wl={metrics['normalized_wl']:.6f}"
            )

        if moved_cells == 0 or total_reward < 1e-5:
            break

    return best_cell_features


def _refine_wirelength_with_bandit_projection(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Run a stronger post-legalization local search for wirelength.

    Small and medium public cases benefit from trying two target models:
      - median target positions (robust for L1-style objectives)
      - median/mean mix (sometimes escapes a different local minimum)

    For the largest public case, the median-only search is both faster and more
    stable, so it is used alone.
    """
    if np is None:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]

    if num_cells <= 300:
        max_passes = 6
        median_candidate = _projected_target_local_search(
            base_cell_features,
            start_cell_features,
            pin_features,
            edge_list,
            target_modes=("median",),
            max_passes=max_passes,
            max_cells_per_pass=num_cells,
            verbose=verbose,
        )
        median_metrics = _calculate_normalized_metrics_fast(median_candidate, pin_features, edge_list)

        mixed_candidate = _projected_target_local_search(
            base_cell_features,
            start_cell_features,
            pin_features,
            edge_list,
            target_modes=("median", "mean"),
            max_passes=max_passes,
            max_cells_per_pass=num_cells,
            verbose=verbose,
        )
        mixed_metrics = _calculate_normalized_metrics_fast(mixed_candidate, pin_features, edge_list)

        if mixed_metrics["overlap_ratio"] == 0.0 and mixed_metrics["normalized_wl"] < median_metrics["normalized_wl"]:
            return mixed_candidate
        return median_candidate

    return _projected_target_local_search(
        base_cell_features,
        start_cell_features,
        pin_features,
        edge_list,
        target_modes=("median",),
        max_passes=4,
        max_cells_per_pass=num_cells,
        verbose=verbose,
    )





def _legal_local_cleanup(base_cell_features, start_cell_features, pin_features, edge_list, verbose=False):
    """Final detailed-placement cleanup that never keeps an illegal candidate."""
    if np is None:
        return start_cell_features

    metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    candidate = _refine_wirelength_by_same_size_assignment(
        base_cell_features, start_cell_features, pin_features, edge_list, verbose=verbose
    )
    candidate = _refine_wirelength_by_pairwise_swaps(
        base_cell_features, candidate, pin_features, edge_list, verbose=verbose
    )
    return candidate


def _reducible_cell_edge_lists(pin_features, edge_list, num_cells):
    """Per-cell edge lists excluding same-cell pin pairs.

    Same-cell pin-to-pin edges are fixed by generated pin offsets: moving the
    cell translates both pins equally. They stay in the final score, but should
    not decide which cells/windows/macros are hot during search.
    """
    if np is None or edge_list.shape[0] == 0:
        return [np.zeros(0, dtype=np.int32) for _ in range(num_cells)]

    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)
    reducible = [[] for _ in range(num_cells)]
    for edge_idx, (src_pin, tgt_pin) in enumerate(edge_np):
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])
        if src_cell == tgt_cell:
            continue
        reducible[src_cell].append(edge_idx)
        reducible[tgt_cell].append(edge_idx)
    return [np.asarray(edges, dtype=np.int32) for edges in reducible]


def _refine_wirelength_by_same_size_assignment(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    max_passes=2,
    max_group_size=140,
    verbose=False,
):
    """Reassign same-size standard cells to existing legal slots.

    Permuting cells among slots with identical dimensions preserves legality, but
    lets us solve a small assignment problem that the continuous optimizer cannot
    express. This is a detailed-placement style independent-set matching pass.
    """
    if np is None or linear_sum_assignment is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    if num_cells < 4 or num_cells > 350:
        return start_cell_features

    pin_cell, incident = _build_incident_edge_lists(pin_features, edge_list, num_cells)
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)

    def local_cost_at(cell_idx, slot_x, slot_y):
        own_pins, other_pins = incident[int(cell_idx)]
        if own_pins.size == 0:
            return 0.0
        other_abs_x = positions[pin_cell[other_pins], 0] + pin_x[other_pins]
        other_abs_y = positions[pin_cell[other_pins], 1] + pin_y[other_pins]
        own_abs_x = slot_x + pin_x[own_pins]
        own_abs_y = slot_y + pin_y[own_pins]
        return float(_smooth_pair_cost_np(np.abs(own_abs_x - other_abs_x), np.abs(own_abs_y - other_abs_y)).sum())

    groups = []
    standard_mask = heights < 1.5
    for width in sorted(set(np.round(widths[standard_mask], 6))):
        group = np.where(standard_mask & (np.abs(widths - width) < 1e-6))[0]
        if group.size >= 3:
            groups.append(group)

    if not groups:
        return start_cell_features

    best_cell_features = start_cell_features.clone()
    best_normalized_wl = start_metrics["normalized_wl"]

    for search_pass in range(max_passes):
        improved_this_pass = False

        for group in groups:
            active_group = group
            if active_group.size > max_group_size:
                hotness = np.asarray(
                    [local_cost_at(cell_idx, positions[cell_idx, 0], positions[cell_idx, 1]) for cell_idx in active_group],
                    dtype=np.float64,
                )
                active_group = active_group[np.argsort(-hotness)[:max_group_size]]

            group_size = active_group.size
            if group_size < 3:
                continue

            slot_positions = positions[active_group].copy()
            cost_matrix = np.empty((group_size, group_size), dtype=np.float64)

            for row, cell_idx in enumerate(active_group):
                own_pins, other_pins = incident[int(cell_idx)]
                if own_pins.size == 0:
                    cost_matrix[row, :] = 0.0
                    continue

                other_abs_x = positions[pin_cell[other_pins], 0] + pin_x[other_pins]
                other_abs_y = positions[pin_cell[other_pins], 1] + pin_y[other_pins]
                own_abs_x = slot_positions[:, 0][None, :] + pin_x[own_pins][:, None]
                own_abs_y = slot_positions[:, 1][None, :] + pin_y[own_pins][:, None]
                dx = np.abs(own_abs_x - other_abs_x[:, None])
                dy = np.abs(own_abs_y - other_abs_y[:, None])
                cost_matrix[row, :] = _smooth_pair_cost_np(dx, dy).sum(axis=0)

            _, col_ind = linear_sum_assignment(cost_matrix)
            if np.all(col_ind == np.arange(group_size)):
                continue

            old_positions = positions[active_group].copy()
            positions[active_group] = slot_positions[col_ind]

            candidate_features = base_cell_features.clone()
            candidate_features[:, 2] = torch.from_numpy(positions[:, 0]).to(candidate_features.dtype)
            candidate_features[:, 3] = torch.from_numpy(positions[:, 1]).to(candidate_features.dtype)
            metrics = _calculate_normalized_metrics_fast(candidate_features, pin_features, edge_list)

            if metrics["overlap_ratio"] == 0.0 and metrics["normalized_wl"] + 1e-12 < best_normalized_wl:
                best_normalized_wl = metrics["normalized_wl"]
                best_cell_features = candidate_features.clone()
                improved_this_pass = True
                if verbose:
                    print(f"  Assignment pass {search_pass}: normalized_wl={best_normalized_wl:.6f}")
            else:
                positions[active_group] = old_positions

        if not improved_this_pass:
            break

    return best_cell_features


def _refine_wirelength_by_pairwise_swaps(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    max_passes=2,
    max_active_cells=90,
    verbose=False,
):
    """Try exact improving swaps among hot cells while preserving legality."""
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    if num_cells < 3 or num_cells > 350:
        return start_cell_features

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)
    src_pins = edge_np[:, 0]
    tgt_pins = edge_np[:, 1]

    cell_edges = [set() for _ in range(num_cells)]
    for edge_idx, (src_pin, tgt_pin) in enumerate(edge_np):
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])
        cell_edges[src_cell].add(edge_idx)
        cell_edges[tgt_cell].add(edge_idx)
    cell_edges = [np.asarray(sorted(edges), dtype=np.int32) for edges in cell_edges]

    def edge_cost(edge_indices):
        if edge_indices.size == 0:
            return 0.0
        src = src_pins[edge_indices]
        tgt = tgt_pins[edge_indices]
        src_x = positions[pin_cell[src], 0] + pin_x[src]
        src_y = positions[pin_cell[src], 1] + pin_y[src]
        tgt_x = positions[pin_cell[tgt], 0] + pin_x[tgt]
        tgt_y = positions[pin_cell[tgt], 1] + pin_y[tgt]
        return float(_smooth_pair_cost_np(np.abs(src_x - tgt_x), np.abs(src_y - tgt_y)).sum())

    def swap_is_legal(cell_i, cell_j):
        for cell_idx in (cell_i, cell_j):
            dx = np.abs(positions[cell_idx, 0] - positions[:, 0])
            dy = np.abs(positions[cell_idx, 1] - positions[:, 1])
            overlap_x = dx < (widths[cell_idx] + widths) * 0.5 - 1e-12
            overlap_y = dy < (heights[cell_idx] + heights) * 0.5 - 1e-12
            overlap = overlap_x & overlap_y
            overlap[cell_idx] = False
            if overlap.any():
                return False
        return True

    reducible_cell_edges = _reducible_cell_edge_lists(pin_features, edge_list, num_cells)
    hotness = np.asarray([edge_cost(edges) for edges in reducible_cell_edges], dtype=np.float64)
    best_cell_features = start_cell_features.clone()
    best_normalized_wl = start_metrics["normalized_wl"]

    for search_pass in range(max_passes):
        moved = 0
        active = np.argsort(-hotness)[: min(num_cells, max_active_cells)]
        pair_order = []
        for left in range(active.size):
            i = int(active[left])
            for right in range(left + 1, active.size):
                j = int(active[right])
                pair_order.append((-(hotness[i] + hotness[j]), i, j))
        pair_order.sort()

        for _, cell_i, cell_j in pair_order:
            if np.allclose(positions[cell_i], positions[cell_j]):
                continue

            edge_union = np.union1d(cell_edges[cell_i], cell_edges[cell_j])
            if edge_union.size == 0:
                continue

            old_i = positions[cell_i].copy()
            old_j = positions[cell_j].copy()
            old_cost = edge_cost(edge_union)

            positions[cell_i] = old_j
            positions[cell_j] = old_i

            if not swap_is_legal(cell_i, cell_j):
                positions[cell_i] = old_i
                positions[cell_j] = old_j
                continue

            new_cost = edge_cost(edge_union)
            if new_cost + 1e-9 < old_cost:
                moved += 1
                hotness[cell_i] = edge_cost(cell_edges[cell_i])
                hotness[cell_j] = edge_cost(cell_edges[cell_j])

                candidate_features = base_cell_features.clone()
                candidate_features[:, 2] = torch.from_numpy(positions[:, 0]).to(candidate_features.dtype)
                candidate_features[:, 3] = torch.from_numpy(positions[:, 1]).to(candidate_features.dtype)
                metrics = _calculate_normalized_metrics_fast(candidate_features, pin_features, edge_list)
                if metrics["overlap_ratio"] == 0.0 and metrics["normalized_wl"] + 1e-12 < best_normalized_wl:
                    best_normalized_wl = metrics["normalized_wl"]
                    best_cell_features = candidate_features.clone()
            else:
                positions[cell_i] = old_i
                positions[cell_j] = old_j

        if verbose:
            print(f"  Swap pass {search_pass}: moved={moved}, normalized_wl={best_normalized_wl:.6f}")
        if moved == 0:
            break

    return best_cell_features
