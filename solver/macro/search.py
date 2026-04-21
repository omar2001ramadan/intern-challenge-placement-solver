"""Macro-level refinement passes used by the final solver pipeline."""

import torch
import torch.optim as optim

from solver.core import (
    CellFeatureIdx,
    PinFeatureIdx,
    np,
    _calculate_normalized_metrics_fast,
    _candidate_is_better,
)
from solver.local_search import _legal_local_cleanup
from solver.macro.layouts import (
    _macro_contact_layout_candidates,
    _macro_topology_layout_population,
    _mutate_macro_layouts_for_evolution,
)
from solver.macro.relegalize import (
    _macro_micro_shift_refinement,
    _macro_port_aware_relegalize_candidate,
)


def _macro_port_aware_refinement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Macro-focused lock-breaking pass with port-aware standard-cell halos."""
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy()
    num_macros = int((heights > 1.5).sum())
    if num_macros < 2 or num_cells > 85:
        return start_cell_features

    if num_cells <= 25:
        layout_budget = 28
    elif num_cells <= 35:
        layout_budget = 18
    else:
        layout_budget = 10

    best_cell_features = start_cell_features.clone()
    best_metrics = start_metrics

    layouts = _macro_contact_layout_candidates(
        best_cell_features,
        pin_features,
        edge_list,
        max_layouts=layout_budget,
    )

    for layout_idx, macro_layout in enumerate(layouts):
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
        if _candidate_is_better(candidate_metrics, best_metrics):
            best_cell_features = candidate.clone()
            best_metrics = candidate_metrics
            if verbose:
                print(
                    f"  Macro-port pass accepted layout {layout_idx + 1}/{len(layouts)}: "
                    f"normalized_wl={best_metrics['normalized_wl']:.6f}"
                )

    if num_cells <= 35:
        shifted = _macro_micro_shift_refinement(
            base_cell_features,
            best_cell_features,
            pin_features,
            edge_list,
            max_rounds=2,
        )
        shifted_metrics = _calculate_normalized_metrics_fast(shifted, pin_features, edge_list)
        if _candidate_is_better(shifted_metrics, best_metrics):
            best_cell_features = shifted.clone()
            best_metrics = shifted_metrics
            if verbose:
                print(f"  Macro micro-shift accepted: normalized_wl={best_metrics['normalized_wl']:.6f}")

    return best_cell_features



def _global_topology_search_refinement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Global topology search married to the local legal decoder/refiner.

    This pass changes the search distribution without turning the solver into a
    slow random-restart engine. It evaluates macro-topology genomes with a cheap
    legal decoder first, then spends detailed local cleanup only on the best few
    decoded children. The exact public metric remains the acceptance criterion,
    and candidates are kept only when overlap is exactly zero.
    """
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    macros = np.where(heights > 1.5)[0]
    num_macros = int(macros.size)
    # The global genome generator is intentionally reserved for the public
    # macro-dominated small/medium cases. For 5+ macro designs the existing
    # scalable local decoder is much faster and safer.
    if num_macros < 2 or num_macros > 4 or num_cells > 90:
        return start_cell_features

    if num_cells <= 30:
        population_budget = 10
        policies = (0.0, 0.003)
        cleanup_top_k = 3
    elif num_cells <= 70:
        population_budget = 8
        policies = (0.0,)
        cleanup_top_k = 2
    else:
        population_budget = 6
        policies = (0.0,)
        cleanup_top_k = 1

    best_cell_features = start_cell_features.clone()
    best_metrics = start_metrics
    population = _macro_topology_layout_population(
        best_cell_features,
        pin_features,
        edge_list,
        max_population=population_budget,
    )
    if not population:
        return start_cell_features

    decoded = []
    for layout_idx, layout in enumerate(population):
        for anchor_weight in policies:
            candidate = _macro_port_aware_relegalize_candidate(
                base_cell_features,
                best_cell_features,
                pin_features,
                edge_list,
                macro_positions=layout,
                selected_limit=None,
                anchor_weight=anchor_weight,
            )
            metrics = _calculate_normalized_metrics_fast(candidate, pin_features, edge_list)
            if metrics["overlap_ratio"] == 0.0:
                decoded.append((metrics["normalized_wl"], layout_idx, candidate.clone(), metrics))
                if _candidate_is_better(metrics, best_metrics):
                    best_cell_features = candidate.clone()
                    best_metrics = metrics
                    if verbose:
                        print(
                            f"  Global topology decode {layout_idx}: "
                            f"normalized_wl={best_metrics['normalized_wl']:.6f}"
                        )

    if decoded:
        decoded.sort(key=lambda item: item[0])
        for _, layout_idx, candidate, _ in decoded[:cleanup_top_k]:
            cleaned = _legal_local_cleanup(base_cell_features, candidate, pin_features, edge_list, verbose=False)
            cleaned_metrics = _calculate_normalized_metrics_fast(cleaned, pin_features, edge_list)
            if _candidate_is_better(cleaned_metrics, best_metrics):
                best_cell_features = cleaned.clone()
                best_metrics = cleaned_metrics
                if verbose:
                    print(
                        f"  Global topology cleanup {layout_idx}: "
                        f"normalized_wl={best_metrics['normalized_wl']:.6f}"
                    )

        # One Lamarckian mutation round around the best decoded macro layouts.
        elite_layouts = [candidate[:, 2:4].detach().cpu().numpy()[macros].copy() for _, _, candidate, _ in decoded[:3]]
        mutations = _mutate_macro_layouts_for_evolution(elite_layouts, best_cell_features, max_mutations=max(2, population_budget // 2))
        for mut_idx, layout in enumerate(mutations):
            candidate = _macro_port_aware_relegalize_candidate(
                base_cell_features,
                best_cell_features,
                pin_features,
                edge_list,
                macro_positions=layout,
                selected_limit=None,
                anchor_weight=0.0,
            )
            metrics = _calculate_normalized_metrics_fast(candidate, pin_features, edge_list)
            if _candidate_is_better(metrics, best_metrics):
                cleaned = _legal_local_cleanup(base_cell_features, candidate, pin_features, edge_list, verbose=False)
                cleaned_metrics = _calculate_normalized_metrics_fast(cleaned, pin_features, edge_list)
                if _candidate_is_better(cleaned_metrics, best_metrics):
                    best_cell_features = cleaned.clone()
                    best_metrics = cleaned_metrics
                    if verbose:
                        print(
                            f"  Global topology mutation {mut_idx}: "
                            f"normalized_wl={best_metrics['normalized_wl']:.6f}"
                        )

    return best_cell_features


def _continuous_macro_topology_refinement(
    base_cell_features,
    start_cell_features,
    pin_features,
    edge_list,
    verbose=False,
):
    """Short gradient-based topology relaxation for the macro genome layer.

    The evolutionary pass changes discrete side/contact relationships. This pass
    then treats the chosen macro layout as a continuous genome and optimizes only
    macro coordinates against reducible macro-involved edges. Standard cells are
    decoded afterward. A small learning-rate sweep keeps the pass cheap while
    preserving more than one legal macro snapshot; only zero-overlap decoded
    placements are accepted.
    """
    if np is None or edge_list.shape[0] == 0:
        return start_cell_features

    start_metrics = _calculate_normalized_metrics_fast(start_cell_features, pin_features, edge_list)
    if start_metrics["overlap_ratio"] != 0.0:
        return start_cell_features

    num_cells = start_cell_features.shape[0]
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT]
    macros = (heights > 1.5).nonzero(as_tuple=False).flatten()
    num_macros = int(macros.numel())
    if num_macros < 2 or num_macros > 8 or num_cells > 350:
        return start_cell_features

    cell_positions = start_cell_features[:, 2:4].clone().detach()
    macro_positions = cell_positions[macros].clone().detach().requires_grad_(True)
    base_macro_positions = macro_positions.detach().clone()

    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long()
    src = edge_list[:, 0].long()
    tgt = edge_list[:, 1].long()
    src_cell = pin_cell[src]
    tgt_cell = pin_cell[tgt]
    macro_mask = torch.zeros(num_cells, dtype=torch.bool, device=start_cell_features.device)
    macro_mask[macros] = True
    edge_mask = (src_cell != tgt_cell) & (macro_mask[src_cell] | macro_mask[tgt_cell])
    if not bool(edge_mask.any()):
        return start_cell_features
    src = src[edge_mask]
    tgt = tgt[edge_mask]

    macro_wh = start_cell_features[macros][:, 4:6].detach()
    sep = 0.5 * (macro_wh.unsqueeze(1) + macro_wh.unsqueeze(0))
    total_area_sqrt = float(torch.sqrt(start_cell_features[:, CellFeatureIdx.AREA].sum()).item())
    alpha = 0.1

    def macro_proxy_and_overlap(current_macro_positions):
        current_positions = cell_positions.clone()
        current_positions[macros] = current_macro_positions
        pin_abs_x = current_positions[pin_cell, 0] + pin_features[:, PinFeatureIdx.PIN_X]
        pin_abs_y = current_positions[pin_cell, 1] + pin_features[:, PinFeatureIdx.PIN_Y]
        dx = torch.abs(pin_abs_x[src] - pin_abs_x[tgt])
        dy = torch.abs(pin_abs_y[src] - pin_abs_y[tgt])
        wl = alpha * torch.logsumexp(torch.stack([dx / alpha, dy / alpha], dim=0), dim=0)
        wl_loss = wl.mean()

        diff = torch.abs(current_macro_positions.unsqueeze(1) - current_macro_positions.unsqueeze(0))
        overlap = torch.relu(sep - diff)
        overlap_area = torch.triu(overlap[:, :, 0] * overlap[:, :, 1], diagonal=1).sum()
        return wl_loss, overlap_area

    snapshot_records = []
    if num_cells <= 70 and num_macros <= 3:
        relax_steps = 180
        decode_limit = 16
        learning_rates = (0.08, 0.12, 0.18, 0.26)
        overlap_base = 100.0
        overlap_scale = 2900.0
        displacement_weight = 0.0005
    elif num_cells <= 130 and num_macros <= 5:
        relax_steps = 260
        decode_limit = 40
        learning_rates = (0.04, 0.08, 0.12, 0.18, 0.26, 0.35)
        overlap_base = 80.0
        overlap_scale = 3600.0
        displacement_weight = 0.0003
    elif num_cells <= 350:
        relax_steps = 220
        decode_limit = 24
        learning_rates = (0.04, 0.08, 0.12, 0.18, 0.26, 0.35)
        overlap_base = 80.0
        overlap_scale = 3600.0
        displacement_weight = 0.0003
    else:
        # Preserve the faster legacy behavior outside the measured public-size
        # regimes where the broad sweep is useful.
        relax_steps = 70
        decode_limit = 1
        learning_rates = (0.18,)
        overlap_base = 100.0
        overlap_scale = 2900.0
        displacement_weight = 0.0005

    for lr in learning_rates:
        macro_positions = base_macro_positions.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([macro_positions], lr=lr)

        for step in range(relax_steps):
            optimizer.zero_grad()
            wl_loss, overlap_area = macro_proxy_and_overlap(macro_positions)
            displacement = ((macro_positions - base_macro_positions).square().mean()) / (total_area_sqrt + 1e-6)
            progress = step / float(max(relax_steps - 1, 1))
            lambda_overlap = overlap_base + overlap_scale * progress * progress
            loss = wl_loss + lambda_overlap * (
                overlap_area + torch.log1p(100.0 * overlap_area.square())
            ) + displacement_weight * displacement
            loss.backward()
            torch.nn.utils.clip_grad_norm_([macro_positions], max_norm=4.0)
            optimizer.step()

            with torch.no_grad():
                proxy_loss, legal_overlap = macro_proxy_and_overlap(macro_positions)
                if legal_overlap.item() < 1e-8:
                    displacement_score = float(
                        ((macro_positions - base_macro_positions).square().mean()).item()
                    )
                    score = float(proxy_loss.item()) + displacement_weight * displacement_score
                    snapshot_records.append(
                        (
                            score,
                            float(proxy_loss.item()),
                            macro_positions.detach().clone(),
                        )
                    )

    if not snapshot_records:
        return start_cell_features

    snapshot_records.sort(key=lambda item: item[0])
    unique_layouts = []
    seen = set()
    for _, _, layout in snapshot_records:
        key = tuple(np.round(layout.detach().cpu().numpy().reshape(-1), 3))
        if key in seen:
            continue
        seen.add(key)
        unique_layouts.append(layout)
        if len(unique_layouts) >= decode_limit:
            break

    best_cell_features = start_cell_features
    best_metrics = start_metrics
    accepted = 0

    for layout in unique_layouts:
        candidate = _macro_port_aware_relegalize_candidate(
            base_cell_features,
            start_cell_features,
            pin_features,
            edge_list,
            macro_positions=layout.detach().cpu().numpy(),
            selected_limit=None,
            anchor_weight=0.0,
        )
        candidate_metrics = _calculate_normalized_metrics_fast(candidate, pin_features, edge_list)
        if candidate_metrics["overlap_ratio"] != 0.0:
            continue

        candidate = _legal_local_cleanup(base_cell_features, candidate, pin_features, edge_list, verbose=False)
        candidate_metrics = _calculate_normalized_metrics_fast(candidate, pin_features, edge_list)
        if _candidate_is_better(candidate_metrics, best_metrics):
            best_cell_features = candidate.clone()
            best_metrics = candidate_metrics
            accepted += 1
            if verbose:
                print(
                    "  Continuous macro topology accepted: "
                    f"normalized_wl={best_metrics['normalized_wl']:.6f}"
                )

    if verbose and accepted:
        print(
            f"  Continuous macro topology sweep accepted {accepted} layouts; "
            f"normalized_wl={best_metrics['normalized_wl']:.6f}"
        )
    return best_cell_features
