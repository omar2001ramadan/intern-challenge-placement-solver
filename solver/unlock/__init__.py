"""Temporary-overlap window unlock and relegalization refinements."""

import torch
import torch.optim as optim

from solver.core import (
    CellFeatureIdx,
    PinFeatureIdx,
    np,
    _calculate_normalized_metrics_fast,
    overlap_repulsion_loss,
    wirelength_attraction_loss,
)
from solver.local_search import (
    _nearest_legal_x_np,
    _nearest_legal_y_np,
    _reducible_cell_edge_lists,
    _smooth_pair_cost_np,
)


def _cell_edge_neighbor_data(pin_features, edge_list, num_cells):
    """Build edge and net-neighbor lists for local detailed placement passes."""
    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)
    src_pins = edge_np[:, 0]
    tgt_pins = edge_np[:, 1]

    cell_edges = [set() for _ in range(num_cells)]
    neighbors = [set() for _ in range(num_cells)]
    for edge_idx, (src_pin, tgt_pin) in enumerate(edge_np):
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])
        cell_edges[src_cell].add(edge_idx)
        cell_edges[tgt_cell].add(edge_idx)
        if src_cell != tgt_cell:
            neighbors[src_cell].add(tgt_cell)
            neighbors[tgt_cell].add(src_cell)

    cell_edges = [np.asarray(sorted(edges), dtype=np.int32) for edges in cell_edges]
    neighbors = [np.asarray(sorted(items), dtype=np.int32) for items in neighbors]
    return pin_cell, src_pins, tgt_pins, cell_edges, neighbors


def _window_unlock_schedule(num_cells):
    """Deterministic budget for the overlap-tolerant LNS refinement."""
    if num_cells <= 25:
        return 40, 16, 40
    if num_cells <= 40:
        return 30, 16, 40
    if num_cells <= 65:
        return 80, 36, 55
    if num_cells <= 90:
        return 80, 40, 55
    if num_cells <= 130:
        return 80, 40, 55
    if num_cells <= 180:
        return 80, 40, 55
    if num_cells <= 350:
        return 80, 40, 55
    return 0, 0, 0


def _overlap_tolerant_window_refinement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Unlock local configurations with temporary overlap, then relegalize.

    This is a large-neighborhood detailed-placement pass aimed at the lock-in
    failure mode: once a solution is legal, purely legal single-cell moves cannot
    cross through another cell to reach a better local ordering. For a handful
    of hot net neighborhoods this routine:
      1. opens a window around a high-cost seed cell,
      2. optimizes the window with a low-to-high overlap penalty so cells may
         briefly pass through each other,
      3. removes the window and reinserts each cell through legal projections,
         choosing projected slots by exact incident wirelength plus a soft
         displacement anchor, and
      4. accepts the candidate only if the public overlap metric is still zero
         and normalized wirelength improves.

    A small UCB controller orders the relegalization operators. It does not
    learn a placer from scratch; it learns which unlock/reinsert ordering has
    recently produced reward on the current design.
    """
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    max_windows, window_size, soft_steps = _window_unlock_schedule(num_cells)
    if max_windows <= 0 or window_size <= 0 or soft_steps <= 1:
        return start_cell_features

    pin_cell, src_pins, tgt_pins, cell_edges, neighbors = _cell_edge_neighbor_data(
        pin_features, edge_list, num_cells
    )
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    areas = start_cell_features[:, CellFeatureIdx.AREA].detach().cpu().numpy().astype(np.float64)

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    best_cell_features = start_cell_features.clone()
    best_normalized_wl = start_metrics["normalized_wl"]
    total_area_sqrt = float(np.sqrt(max(areas.sum(), 1.0)))

    def edge_cost(edge_indices, pos):
        if edge_indices.size == 0:
            return 0.0
        src = src_pins[edge_indices]
        tgt = tgt_pins[edge_indices]
        src_x = pos[pin_cell[src], 0] + pin_x[src]
        src_y = pos[pin_cell[src], 1] + pin_y[src]
        tgt_x = pos[pin_cell[tgt], 0] + pin_x[tgt]
        tgt_y = pos[pin_cell[tgt], 1] + pin_y[tgt]
        return float(_smooth_pair_cost_np(np.abs(src_x - tgt_x), np.abs(src_y - tgt_y)).sum())

    def local_cost_at(cell_idx, cand_x, cand_y, pos):
        old_position = pos[cell_idx].copy()
        pos[cell_idx, 0] = cand_x
        pos[cell_idx, 1] = cand_y
        cost = edge_cost(cell_edges[cell_idx], pos)
        pos[cell_idx] = old_position
        return cost

    reducible_cell_edges = _reducible_cell_edge_lists(pin_features, edge_list, num_cells)
    hotness = np.asarray([edge_cost(reducible_cell_edges[cell_idx], positions) for cell_idx in range(num_cells)])
    seed_order = np.argsort(-hotness)[: max_windows * 2]

    op_names = ("hot", "area", "travel", "small")
    bandit_counts = np.ones(len(op_names), dtype=np.float64)
    bandit_rewards = np.zeros(len(op_names), dtype=np.float64)
    attempted_windows = set()
    accepted = 0

    for seed in seed_order:
        if len(attempted_windows) >= max_windows:
            break
        seed = int(seed)

        window_cells = {seed}
        direct_neighbors = sorted((int(x) for x in neighbors[seed]), key=lambda x: -hotness[x])
        for cell_idx in direct_neighbors[: max(0, window_size - 1)]:
            window_cells.add(cell_idx)

        if len(window_cells) < window_size:
            neighbor_pool = set()
            for cell_idx in direct_neighbors[:8]:
                neighbor_pool.update(int(x) for x in neighbors[cell_idx])
            neighbor_pool.difference_update(window_cells)
            for cell_idx in sorted(neighbor_pool, key=lambda x: -hotness[x])[: window_size - len(window_cells)]:
                window_cells.add(cell_idx)

        if len(window_cells) < window_size:
            distance_to_seed = np.linalg.norm(positions - positions[seed], axis=1)
            for cell_idx in np.argsort(distance_to_seed):
                if len(window_cells) >= window_size:
                    break
                window_cells.add(int(cell_idx))

        window = np.asarray(sorted(window_cells), dtype=np.int64)
        if window.size < 3:
            continue
        window_key = tuple(window.tolist())
        if window_key in attempted_windows:
            continue
        attempted_windows.add(window_key)

        base_positions = positions.copy()
        anchor = base_positions[window].copy()

        # Soft illegal phase: allow cells in the window to pass through one
        # another early, then turn overlap pressure back up before legalization.
        window_t = torch.from_numpy(window).long()
        anchor_t = torch.tensor(anchor, dtype=base_cell_features.dtype)
        wpos = anchor_t.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([wpos], lr=0.15)
        base_template = base_cell_features.clone()
        fixed_positions_t = torch.from_numpy(base_positions).to(dtype=base_template.dtype)

        for step in range(soft_steps):
            optimizer.zero_grad()
            current_positions = fixed_positions_t.clone()
            current_positions[window_t] = wpos
            current_features = base_template.clone()
            current_features[:, 2:4] = current_positions

            wl_loss = wirelength_attraction_loss(current_features, pin_features, edge_list)
            overlap_loss = overlap_repulsion_loss(current_features, pin_features, edge_list)
            displacement_loss = ((wpos - anchor_t).square().mean()) / (total_area_sqrt + 1e-6)
            progress = step / float(max(soft_steps - 1, 1))
            lambda_overlap = 0.02 + 3.0 * progress * progress
            total_loss = wl_loss + lambda_overlap * overlap_loss + 0.01 * displacement_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([wpos], max_norm=2.0)
            optimizer.step()

        soft_targets = wpos.detach().cpu().numpy().astype(np.float64)
        travel = np.linalg.norm(soft_targets - anchor, axis=1)

        order_candidates = {
            "hot": window[np.argsort(-hotness[window])],
            "area": window[np.argsort(-areas[window])],
            "travel": window[np.argsort(-travel)],
            "small": window[np.argsort(areas[window])],
        }

        total_trials = bandit_counts.sum()
        ucb = (bandit_rewards / bandit_counts) + 0.20 * np.sqrt(np.log(total_trials + 1.0) / bandit_counts)
        op_order = np.argsort(-ucb)

        best_window_candidate = None
        best_window_wl = best_normalized_wl
        best_window_op = None

        for op_idx in op_order:
            op_name = op_names[int(op_idx)]
            order = order_candidates[op_name]

            # Remove the whole window, then greedily reinsert cells. This is
            # the key escape hatch: cells are not forced to preserve their old
            # legal ordering while the local window is being repaired.
            candidate_positions = base_positions.copy()
            far_coordinate = 1e8
            for offset, cell_idx in enumerate(window):
                candidate_positions[cell_idx, 0] = far_coordinate + 1000.0 * offset
                candidate_positions[cell_idx, 1] = far_coordinate + 1000.0 * offset

            for cell_idx in (int(x) for x in order):
                window_offset = int(np.where(window == cell_idx)[0][0])
                target_x, target_y = soft_targets[window_offset]
                anchor_x, anchor_y = anchor[window_offset]

                candidates = []
                for frac in (1.0, 0.75, 0.50, 0.25, 0.0):
                    probe_x = anchor_x + frac * (target_x - anchor_x)
                    probe_y = anchor_y + frac * (target_y - anchor_y)

                    cand_x = _nearest_legal_x_np(cell_idx, probe_x, probe_y, candidate_positions, widths, heights)
                    cand_y = _nearest_legal_y_np(cell_idx, probe_y, cand_x, candidate_positions, widths, heights)
                    candidates.append((cand_x, cand_y))

                    cand_y = _nearest_legal_y_np(cell_idx, probe_y, probe_x, candidate_positions, widths, heights)
                    cand_x = _nearest_legal_x_np(cell_idx, probe_x, cand_y, candidate_positions, widths, heights)
                    candidates.append((cand_x, cand_y))

                best_insert = None
                for cand_x, cand_y in candidates:
                    local_cost = local_cost_at(cell_idx, cand_x, cand_y, candidate_positions)
                    anchor_cost = 0.005 * ((cand_x - anchor_x) ** 2 + (cand_y - anchor_y) ** 2)
                    score = local_cost + anchor_cost
                    if best_insert is None or score < best_insert[0]:
                        best_insert = (score, cand_x, cand_y)

                candidate_positions[cell_idx, 0] = best_insert[1]
                candidate_positions[cell_idx, 1] = best_insert[2]

            candidate_features = base_cell_features.clone()
            candidate_features[:, 2] = torch.from_numpy(candidate_positions[:, 0]).to(candidate_features.dtype)
            candidate_features[:, 3] = torch.from_numpy(candidate_positions[:, 1]).to(candidate_features.dtype)
            metrics = _calculate_normalized_metrics_fast(candidate_features, pin_features, edge_list)

            reward = max(0.0, best_normalized_wl - metrics["normalized_wl"])
            bandit_counts[int(op_idx)] += 1.0
            bandit_rewards[int(op_idx)] += reward

            if metrics["overlap_ratio"] == 0.0 and metrics["normalized_wl"] + 1e-12 < best_window_wl:
                best_window_wl = metrics["normalized_wl"]
                best_window_candidate = (candidate_features.clone(), candidate_positions.copy())
                best_window_op = int(op_idx)

        if best_window_candidate is not None:
            reward = best_normalized_wl - best_window_wl
            if best_window_op is not None:
                bandit_rewards[best_window_op] += reward
            best_cell_features, positions = best_window_candidate
            best_normalized_wl = best_window_wl
            accepted += 1
            if verbose:
                print(
                    f"  Unlock window {len(attempted_windows)}/{max_windows}: "
                    f"accepted size={window.size}, normalized_wl={best_normalized_wl:.6f}"
                )

    if verbose and accepted:
        print(f"  Overlap-tolerant unlock accepted {accepted} windows; normalized_wl={best_normalized_wl:.6f}")

    return best_cell_features
