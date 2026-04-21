"""Macro-aware standard-cell relegalization routines."""

import torch

from solver.core import (
    CellFeatureIdx,
    PinFeatureIdx,
    np,
    _calculate_normalized_metrics_fast,
    _candidate_is_better,
)
from solver.local_search import (
    _nearest_legal_x_np,
    _nearest_legal_y_np,
    _smooth_pair_cost_np,
)


def _macro_port_aware_relegalize_candidate(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    macro_positions=None,
    selected_limit=None,
    anchor_weight=0.0,
):
    """Remove selected standard cells and reinsert them into macro halo slots."""
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    anchor_positions = positions.copy()
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    macro_mask = heights > 1.5
    macros = np.where(macro_mask)[0]
    std_cells = np.where(~macro_mask)[0]

    if macros.size < 1 or std_cells.size < 1:
        return start_cell_features
    if macro_positions is not None:
        positions[macros] = macro_positions

    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)
    src_pins = edge_np[:, 0]
    tgt_pins = edge_np[:, 1]

    own_pins = [[] for _ in range(num_cells)]
    other_pins = [[] for _ in range(num_cells)]
    reducible_edges = [[] for _ in range(num_cells)]
    macro_degree = np.zeros(num_cells, dtype=np.float64)

    for edge_idx, (src_pin, tgt_pin) in enumerate(edge_np):
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])
        own_pins[src_cell].append(src_pin)
        other_pins[src_cell].append(tgt_pin)
        if src_cell != tgt_cell:
            own_pins[tgt_cell].append(tgt_pin)
            other_pins[tgt_cell].append(src_pin)
            reducible_edges[src_cell].append(edge_idx)
            reducible_edges[tgt_cell].append(edge_idx)
            if macro_mask[src_cell] or macro_mask[tgt_cell]:
                macro_degree[src_cell] += 1.0
                macro_degree[tgt_cell] += 1.0

    own_pins = [np.asarray(items, dtype=np.int64) for items in own_pins]
    other_pins = [np.asarray(items, dtype=np.int64) for items in other_pins]

    def edge_cost(edge_indices, pos):
        if len(edge_indices) == 0:
            return 0.0
        edge_indices = np.asarray(edge_indices, dtype=np.int64)
        src = src_pins[edge_indices]
        tgt = tgt_pins[edge_indices]
        src_x = pos[pin_cell[src], 0] + pin_x[src]
        src_y = pos[pin_cell[src], 1] + pin_y[src]
        tgt_x = pos[pin_cell[tgt], 0] + pin_x[tgt]
        tgt_y = pos[pin_cell[tgt], 1] + pin_y[tgt]
        return float(_smooth_pair_cost_np(np.abs(src_x - tgt_x), np.abs(src_y - tgt_y)).sum())

    reducible_hotness = np.asarray(
        [edge_cost(reducible_edges[cell_idx], positions) for cell_idx in range(num_cells)],
        dtype=np.float64,
    )

    selected = std_cells.copy()
    if selected_limit is not None and selected.size > selected_limit:
        selected = selected[
            np.argsort(-(reducible_hotness[selected] + 0.25 * macro_degree[selected]))[:selected_limit]
        ]

    legal_positions = positions.copy()
    score_positions = positions.copy()
    far_coordinate = 1e7
    for offset, cell_idx in enumerate(selected):
        legal_positions[cell_idx, 0] = far_coordinate + 1000.0 * offset
        legal_positions[cell_idx, 1] = far_coordinate + 1000.0 * offset
        score_positions[cell_idx] = positions[cell_idx]

    def local_cost_at(cell_idx, cand_x, cand_y):
        own = own_pins[cell_idx]
        other = other_pins[cell_idx]
        if own.size == 0:
            return 0.0
        other_abs_x = score_positions[pin_cell[other], 0] + pin_x[other]
        other_abs_y = score_positions[pin_cell[other], 1] + pin_y[other]
        own_abs_x = cand_x + pin_x[own]
        own_abs_y = cand_y + pin_y[own]
        return float(_smooth_pair_cost_np(np.abs(own_abs_x - other_abs_x), np.abs(own_abs_y - other_abs_y)).sum())

    def slot_targets_for_cell(cell_idx):
        targets = []
        all_target_x = []
        all_target_y = []
        per_macro_targets = {}

        for own_pin, other_pin in zip(own_pins[cell_idx], other_pins[cell_idx]):
            other_cell = int(pin_cell[other_pin])
            if other_cell == cell_idx:
                continue
            other_abs_x = score_positions[other_cell, 0] + pin_x[other_pin]
            other_abs_y = score_positions[other_cell, 1] + pin_y[other_pin]
            ideal_x = other_abs_x - pin_x[own_pin]
            ideal_y = other_abs_y - pin_y[own_pin]
            all_target_x.append(ideal_x)
            all_target_y.append(ideal_y)
            if macro_mask[other_cell]:
                per_macro_targets.setdefault(other_cell, ([], []))
                per_macro_targets[other_cell][0].append(ideal_x)
                per_macro_targets[other_cell][1].append(ideal_y)

        if all_target_x:
            targets.append((float(np.median(all_target_x)), float(np.median(all_target_y))))
            targets.append((float(np.mean(all_target_x)), float(np.mean(all_target_y))))

        for macro_idx, (target_xs, target_ys) in per_macro_targets.items():
            macro_x, macro_y = score_positions[macro_idx]
            macro_w = widths[macro_idx]
            macro_h = heights[macro_idx]
            cell_w = widths[cell_idx]
            cell_h = heights[cell_idx]
            median_x = float(np.median(target_xs))
            median_y = float(np.median(target_ys))
            targets.extend(
                [
                    (macro_x + 0.5 * (macro_w + cell_w) + 1e-4, median_y),
                    (macro_x - 0.5 * (macro_w + cell_w) - 1e-4, median_y),
                    (median_x, macro_y + 0.5 * (macro_h + cell_h) + 1e-4),
                    (median_x, macro_y - 0.5 * (macro_h + cell_h) - 1e-4),
                ]
            )

        targets.append(tuple(anchor_positions[cell_idx]))
        targets.append(tuple(positions[cell_idx]))
        return targets

    insertion_order = selected[
        np.argsort(-(reducible_hotness[selected] + 0.25 * macro_degree[selected]))
    ]

    for cell_idx in (int(x) for x in insertion_order):
        anchor_x, anchor_y = anchor_positions[cell_idx]
        best_slot = None
        seen = set()

        for target_x, target_y in slot_targets_for_cell(cell_idx):
            for fraction in (1.0, 0.8, 0.5, 0.2, 0.0):
                probe_x = anchor_x + fraction * (target_x - anchor_x)
                probe_y = anchor_y + fraction * (target_y - anchor_y)

                cand_x = _nearest_legal_x_np(
                    cell_idx, probe_x, probe_y, legal_positions, widths, heights, margin=1e-5
                )
                cand_y = _nearest_legal_y_np(
                    cell_idx, probe_y, cand_x, legal_positions, widths, heights, margin=1e-5
                )
                candidates = [(cand_x, cand_y)]

                cand_y = _nearest_legal_y_np(
                    cell_idx, probe_y, probe_x, legal_positions, widths, heights, margin=1e-5
                )
                cand_x = _nearest_legal_x_np(
                    cell_idx, probe_x, cand_y, legal_positions, widths, heights, margin=1e-5
                )
                candidates.append((cand_x, cand_y))

                for cand_x, cand_y in candidates:
                    key = (round(cand_x, 4), round(cand_y, 4))
                    if key in seen:
                        continue
                    seen.add(key)
                    score = local_cost_at(cell_idx, cand_x, cand_y)
                    if anchor_weight > 0.0:
                        score += anchor_weight * ((cand_x - anchor_x) ** 2 + (cand_y - anchor_y) ** 2)
                    if best_slot is None or score < best_slot[0]:
                        best_slot = (score, cand_x, cand_y)

        if best_slot is None:
            legal_positions[cell_idx] = anchor_positions[cell_idx]
            score_positions[cell_idx] = anchor_positions[cell_idx]
        else:
            legal_positions[cell_idx, 0] = best_slot[1]
            legal_positions[cell_idx, 1] = best_slot[2]
            score_positions[cell_idx, 0] = best_slot[1]
            score_positions[cell_idx, 1] = best_slot[2]

    candidate_features = base_cell_features.clone()
    candidate_features[:, 2] = torch.from_numpy(legal_positions[:, 0]).to(candidate_features.dtype)
    candidate_features[:, 3] = torch.from_numpy(legal_positions[:, 1]).to(candidate_features.dtype)
    return candidate_features



def _macro_micro_shift_refinement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    max_rounds=2,
):
    """Tiny macro-coordinate hill climb followed by port-aware relegalization.

    This is deliberately not a general continuous optimizer. It tests a small
    set of sub-cell-ish macro shifts and scores the fully relegalized placement.
    It helps cases where the macro topology is right but the legal contact
    point is one or two units away from a better pin-cloud alignment.
    """
    if np is None:
        return start_cell_features

    best_cell_features = start_cell_features.clone()
    best_metrics = _calculate_normalized_metrics_fast(best_cell_features, pin_features, edge_list)
    if best_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    for _ in range(max_rounds):
        heights = best_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
        widths = best_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
        positions = best_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64)
        macros = np.where(heights > 1.5)[0]
        if macros.size < 2:
            break

        round_best_features = best_cell_features
        round_best_metrics = best_metrics
        round_improved = False

        for macro_offset, macro_idx in enumerate(macros):
            # Keep this list small. The relegalized candidate is the expensive
            # part, and the public small macro cases need only fine nudges.
            for step in (1.0, 2.0, 0.5):
                for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                    macro_layout = positions[macros].copy()
                    macro_layout[macro_offset, 0] += dx
                    macro_layout[macro_offset, 1] += dy

                    legal_macros = True
                    for left in range(macros.size):
                        i = int(macros[left])
                        for right in range(left + 1, macros.size):
                            j = int(macros[right])
                            sep_x = 0.5 * (widths[i] + widths[j])
                            sep_y = 0.5 * (heights[i] + heights[j])
                            if (
                                abs(macro_layout[left, 0] - macro_layout[right, 0]) < sep_x - 1e-7
                                and abs(macro_layout[left, 1] - macro_layout[right, 1]) < sep_y - 1e-7
                            ):
                                legal_macros = False
                                break
                        if not legal_macros:
                            break
                    if not legal_macros:
                        continue

                    candidate = _macro_port_aware_relegalize_candidate(
                        base_cell_features,
                        best_cell_features,
                        pin_features,
                        edge_list,
                        macro_positions=macro_layout,
                        selected_limit=None,
                        anchor_weight=0.0,
                    )
                    candidate_metrics = _calculate_normalized_metrics_fast(candidate, pin_features, edge_list)
                    if _candidate_is_better(candidate_metrics, round_best_metrics):
                        round_best_features = candidate.clone()
                        round_best_metrics = candidate_metrics
                        round_improved = True

        if not round_improved:
            break

        best_cell_features = round_best_features
        best_metrics = round_best_metrics

    return best_cell_features

