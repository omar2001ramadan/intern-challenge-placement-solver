"""Macro layout and topology-population generators."""

from solver.core import CellFeatureIdx, PinFeatureIdx, np
from solver.local_search import _smooth_pair_cost_np


def _macro_contact_layout_candidates(start_cell_features, pin_features, edge_list, max_layouts=32):
    """Generate legal macro-cluster contact layouts around the current solution."""
    if np is None:
        return []

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64)
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    macros = np.where(heights > 1.5)[0]
    num_macros = int(macros.size)
    if num_macros < 2 or num_macros > 4:
        return [positions[macros].copy()]

    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)

    pair_offsets = {}
    for parent in macros:
        for child in macros:
            if parent == child:
                continue
            dx_values = []
            dy_values = []
            for src_pin, tgt_pin in edge_np:
                src_cell = int(pin_cell[src_pin])
                tgt_cell = int(pin_cell[tgt_pin])
                if src_cell == int(parent) and tgt_cell == int(child):
                    dx_values.append(pin_x[src_pin] - pin_x[tgt_pin])
                    dy_values.append(pin_y[src_pin] - pin_y[tgt_pin])
                elif src_cell == int(child) and tgt_cell == int(parent):
                    dx_values.append(pin_x[tgt_pin] - pin_x[src_pin])
                    dy_values.append(pin_y[tgt_pin] - pin_y[src_pin])
            pair_offsets[(int(parent), int(child))] = (
                float(np.median(dx_values)) if dx_values else 0.0,
                float(np.median(dy_values)) if dy_values else 0.0,
            )

    current_macro_positions = positions[macros].copy()

    def layout_is_legal(macro_positions):
        for left in range(num_macros):
            i = int(macros[left])
            for right in range(left + 1, num_macros):
                j = int(macros[right])
                dx = abs(macro_positions[left, 0] - macro_positions[right, 0])
                dy = abs(macro_positions[left, 1] - macro_positions[right, 1])
                if (
                    dx < 0.5 * (widths[i] + widths[j]) - 1e-7
                    and dy < 0.5 * (heights[i] + heights[j]) - 1e-7
                ):
                    return False
        return True

    def relative_offset(parent, child, side, spacing_scale=1.0, align_scale=1.0):
        align_dx, align_dy = pair_offsets[(int(parent), int(child))]
        sep_x = 0.5 * (widths[parent] + widths[child]) * spacing_scale + 1e-4
        sep_y = 0.5 * (heights[parent] + heights[child]) * spacing_scale + 1e-4
        if side == "R":
            return np.asarray([sep_x, align_scale * align_dy], dtype=np.float64)
        if side == "L":
            return np.asarray([-sep_x, align_scale * align_dy], dtype=np.float64)
        if side == "T":
            return np.asarray([align_scale * align_dx, sep_y], dtype=np.float64)
        return np.asarray([align_scale * align_dx, -sep_y], dtype=np.float64)

    layouts = [current_macro_positions]

    centroid = current_macro_positions.mean(axis=0)
    centered = current_macro_positions - centroid
    transforms = (
        np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
        np.asarray([[-1.0, 0.0], [0.0, -1.0]]),
    )
    for transform in transforms:
        transformed = centered @ transform.T + centroid
        if layout_is_legal(transformed):
            layouts.append(transformed)

    sides = ("R", "L", "T", "B")
    macro_ids = [int(x) for x in macros]
    permutations = list(__import__("itertools").permutations(macro_ids))

    for spacing_scale in (1.0, 1.04, 1.10):
        for align_scale in (0.0, 0.5, 1.0):
            for perm in permutations:
                partial_layouts = [{perm[0]: np.asarray([0.0, 0.0], dtype=np.float64)}]
                for child in perm[1:]:
                    next_layouts = []
                    for partial in partial_layouts:
                        for parent in tuple(partial.keys()):
                            for side in sides:
                                candidate = {key: value.copy() for key, value in partial.items()}
                                candidate[child] = (
                                    partial[parent]
                                    + relative_offset(parent, child, side, spacing_scale, align_scale)
                                )

                                keys = tuple(candidate.keys())
                                legal = True
                                for left in range(len(keys)):
                                    i = keys[left]
                                    for right in range(left + 1, len(keys)):
                                        j = keys[right]
                                        dx = abs(candidate[i][0] - candidate[j][0])
                                        dy = abs(candidate[i][1] - candidate[j][1])
                                        if (
                                            dx < 0.5 * (widths[i] + widths[j]) - 1e-7
                                            and dy < 0.5 * (heights[i] + heights[j]) - 1e-7
                                        ):
                                            legal = False
                                            break
                                    if not legal:
                                        break
                                if legal:
                                    next_layouts.append(candidate)
                    partial_layouts = next_layouts[:240]

                for partial in partial_layouts[:240]:
                    raw = np.zeros((num_macros, 2), dtype=np.float64)
                    for macro_offset, macro_idx in enumerate(macros):
                        raw[macro_offset] = partial[int(macro_idx)]

                    shifts = [current_macro_positions.mean(axis=0) - raw.mean(axis=0)]
                    for macro_offset, macro_idx in enumerate(macros):
                        shifts.append(positions[int(macro_idx)] - raw[macro_offset])

                    for shift in shifts:
                        shifted = raw + shift
                        if layout_is_legal(shifted):
                            layouts.append(shifted)

    unique = []
    seen = set()
    for layout in layouts:
        key = tuple(np.round(layout.reshape(-1), 1))
        if key in seen:
            continue
        seen.add(key)
        displacement = float(np.square(layout - current_macro_positions).sum())
        unique.append((displacement, layout.copy()))

    unique.sort(key=lambda item: item[0])
    return [layout for _, layout in unique[:max_layouts]]




def _macro_layout_proxy_cost_np(layout, start_cell_features, pin_features, edge_list):
    """Cheap score for macro topology genomes before full decoding.

    Same-cell edges are ignored because they are invariant under translation.
    Standard cells remain at their current positions, so this is only a basin
    selector; the exact public metric is still used after decoding/legalization.
    """
    if np is None or edge_list.shape[0] == 0:
        return 0.0

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64).copy()
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    macros = np.where(heights > 1.5)[0]
    if macros.size == 0:
        return 0.0
    positions[macros] = layout

    # Hard macro legality guard.
    for left in range(macros.size):
        i = int(macros[left])
        for right in range(left + 1, macros.size):
            j = int(macros[right])
            if (
                abs(positions[i, 0] - positions[j, 0]) < 0.5 * (widths[i] + widths[j]) - 1e-7
                and abs(positions[i, 1] - positions[j, 1]) < 0.5 * (heights[i] + heights[j]) - 1e-7
            ):
                return float("inf")

    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)
    src = edge_np[:, 0]
    tgt = edge_np[:, 1]
    src_cell = pin_cell[src]
    tgt_cell = pin_cell[tgt]
    reducible = src_cell != tgt_cell
    if not reducible.any():
        return 0.0

    # Emphasize macro-involved reducible edges because the global topology pass
    # exists to choose a different basin, not to polish standard-cell details.
    macro_mask = heights > 1.5
    macro_edge = macro_mask[src_cell] | macro_mask[tgt_cell]
    weights = np.ones(edge_np.shape[0], dtype=np.float64)
    weights[macro_edge] = 2.5
    weights = weights[reducible]
    src = src[reducible]
    tgt = tgt[reducible]

    src_x = positions[pin_cell[src], 0] + pin_x[src]
    src_y = positions[pin_cell[src], 1] + pin_y[src]
    tgt_x = positions[pin_cell[tgt], 0] + pin_x[tgt]
    tgt_y = positions[pin_cell[tgt], 1] + pin_y[tgt]
    return float((_smooth_pair_cost_np(np.abs(src_x - tgt_x), np.abs(src_y - tgt_y)) * weights).sum())


def _macro_topology_layout_population(start_cell_features, pin_features, edge_list, max_population=36):
    """Generate a diverse population of macro-topology genomes.

    This is the global-search layer. A genome is just a legal macro layout; the
    decoder below decides how standard cells should be reinserted around it.
    Layouts are ranked by a reducible-edge proxy rather than by distance from the
    current legal placement, so far-away basins can survive long enough to be
    decoded and evaluated by the true metric.
    """
    if np is None:
        return []

    positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64)
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    macros = np.where(heights > 1.5)[0]
    num_macros = int(macros.size)
    if num_macros < 2 or num_macros > 5:
        return [positions[macros].copy()] if num_macros else []

    current = positions[macros].copy()
    centroid = current.mean(axis=0)

    def is_legal(layout):
        for left in range(num_macros):
            i = int(macros[left])
            for right in range(left + 1, num_macros):
                j = int(macros[right])
                if (
                    abs(layout[left, 0] - layout[right, 0]) < 0.5 * (widths[i] + widths[j]) - 1e-7
                    and abs(layout[left, 1] - layout[right, 1]) < 0.5 * (heights[i] + heights[j]) - 1e-7
                ):
                    return False
        return True

    layouts = [current]

    # Reuse the local contact generator, but ask for more than the old local
    # pass used. These are then reranked globally rather than by displacement.
    try:
        layouts.extend(_macro_contact_layout_candidates(start_cell_features, pin_features, edge_list, max_layouts=(120 if max_population > 8 else 70)))
    except Exception:
        pass

    # Add simple global transforms around the macro centroid.
    centered = current - centroid
    transforms = (
        np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        np.asarray([[-1.0, 0.0], [0.0, -1.0]]),
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
    )
    for transform in transforms:
        transformed = centered @ transform.T + centroid
        if is_legal(transformed):
            layouts.append(transformed)

    # Exhaustive side-contact layouts for tiny macro counts. This is the true
    # topology search: the current solution is not assumed to be the right tree.
    pin_cell = pin_features[:, PinFeatureIdx.CELL_IDX].long().detach().cpu().numpy()
    pin_x = pin_features[:, PinFeatureIdx.PIN_X].detach().cpu().numpy().astype(np.float64)
    pin_y = pin_features[:, PinFeatureIdx.PIN_Y].detach().cpu().numpy().astype(np.float64)
    edge_np = edge_list.detach().cpu().numpy().astype(np.int64)

    pair_stats = {}
    for parent in macros:
        for child in macros:
            if parent == child:
                continue
            dx_vals = []
            dy_vals = []
            for src_pin, tgt_pin in edge_np:
                src_cell = int(pin_cell[src_pin])
                tgt_cell = int(pin_cell[tgt_pin])
                if src_cell == int(parent) and tgt_cell == int(child):
                    dx_vals.append(pin_x[src_pin] - pin_x[tgt_pin])
                    dy_vals.append(pin_y[src_pin] - pin_y[tgt_pin])
                elif src_cell == int(child) and tgt_cell == int(parent):
                    dx_vals.append(pin_x[tgt_pin] - pin_x[src_pin])
                    dy_vals.append(pin_y[tgt_pin] - pin_y[src_pin])
            if dx_vals:
                pair_stats[(int(parent), int(child))] = (
                    float(np.median(dx_vals)),
                    float(np.median(dy_vals)),
                    float(np.mean(dx_vals)),
                    float(np.mean(dy_vals)),
                )
            else:
                pair_stats[(int(parent), int(child))] = (0.0, 0.0, 0.0, 0.0)

    def relative_offset(parent, child, side, spacing_scale, align_scale, use_mean=False):
        med_dx, med_dy, mean_dx, mean_dy = pair_stats[(int(parent), int(child))]
        align_dx = mean_dx if use_mean else med_dx
        align_dy = mean_dy if use_mean else med_dy
        sep_x = 0.5 * (widths[parent] + widths[child]) * spacing_scale + 1e-4
        sep_y = 0.5 * (heights[parent] + heights[child]) * spacing_scale + 1e-4
        if side == "R":
            return np.asarray([sep_x, align_scale * align_dy], dtype=np.float64)
        if side == "L":
            return np.asarray([-sep_x, align_scale * align_dy], dtype=np.float64)
        if side == "T":
            return np.asarray([align_scale * align_dx, sep_y], dtype=np.float64)
        return np.asarray([align_scale * align_dx, -sep_y], dtype=np.float64)

    import itertools
    macro_ids = [int(x) for x in macros]
    # Beam width keeps this deterministic and bounded for 4-5 macro cases.
    beam_width = 72 if num_macros <= 3 else 24
    sides = ("R", "L", "T", "B")
    spacing_scales = (1.0, 1.04, 1.12)
    align_scales = (0.0, 0.5, 1.0)

    for spacing_scale in spacing_scales:
        for align_scale in align_scales:
            for use_mean in (False, True):
                for perm in itertools.permutations(macro_ids):
                    partials = [{perm[0]: np.asarray([0.0, 0.0], dtype=np.float64)}]
                    for child in perm[1:]:
                        next_partials = []
                        for partial in partials:
                            for parent in tuple(partial.keys()):
                                for side in sides:
                                    candidate = {k: v.copy() for k, v in partial.items()}
                                    candidate[child] = partial[parent] + relative_offset(
                                        parent, child, side, spacing_scale, align_scale, use_mean=use_mean
                                    )
                                    keys = tuple(candidate.keys())
                                    legal = True
                                    for left in range(len(keys)):
                                        i = keys[left]
                                        for right in range(left + 1, len(keys)):
                                            j = keys[right]
                                            if (
                                                abs(candidate[i][0] - candidate[j][0]) < 0.5 * (widths[i] + widths[j]) - 1e-7
                                                and abs(candidate[i][1] - candidate[j][1]) < 0.5 * (heights[i] + heights[j]) - 1e-7
                                            ):
                                                legal = False
                                                break
                                        if not legal:
                                            break
                                    if legal:
                                        next_partials.append(candidate)
                        if len(next_partials) > beam_width:
                            # Use a cheap compactness proxy during construction.
                            next_partials.sort(
                                key=lambda partial: sum(float(np.dot(v, v)) for v in partial.values())
                            )
                            next_partials = next_partials[:beam_width]
                        partials = next_partials

                    for partial in partials:
                        raw = np.zeros((num_macros, 2), dtype=np.float64)
                        for macro_offset, macro_idx in enumerate(macros):
                            raw[macro_offset] = partial[int(macro_idx)]

                        shifts = [centroid - raw.mean(axis=0)]
                        for macro_offset, macro_idx in enumerate(macros):
                            shifts.append(positions[int(macro_idx)] - raw[macro_offset])
                        for shift in shifts:
                            shifted = raw + shift
                            if is_legal(shifted):
                                layouts.append(shifted)

    # Add line and grid layouts ordered by macro reducible degree. These are
    # useful when the contact tree generator over-commits to a bad side choice.
    reducible_degree = np.zeros(start_cell_features.shape[0], dtype=np.float64)
    for src_pin, tgt_pin in edge_np:
        src_cell = int(pin_cell[src_pin])
        tgt_cell = int(pin_cell[tgt_pin])
        if src_cell != tgt_cell:
            reducible_degree[src_cell] += 1.0
            reducible_degree[tgt_cell] += 1.0
    order = [int(x) for x in macros[np.argsort(-reducible_degree[macros])]]
    for axis in (0, 1):
        for direction in (-1.0, 1.0):
            raw = np.zeros((num_macros, 2), dtype=np.float64)
            cursor = 0.0
            for rank, macro_idx in enumerate(order):
                macro_offset = int(np.where(macros == macro_idx)[0][0])
                if rank == 0:
                    raw[macro_offset] = 0.0
                else:
                    prev_idx = order[rank - 1]
                    gap = 0.5 * ((widths[prev_idx] + widths[macro_idx]) if axis == 0 else (heights[prev_idx] + heights[macro_idx])) + 1e-4
                    cursor += gap
                    raw[macro_offset, axis] = direction * cursor
            raw += centroid - raw.mean(axis=0)
            if is_legal(raw):
                layouts.append(raw)

    # Rerank by topology proxy plus a tiny displacement term to break ties while
    # still allowing basin-changing candidates.
    current_flat = current.reshape(-1)
    unique = []
    seen = set()
    for layout in layouts:
        if not is_legal(layout):
            continue
        key = tuple(np.round(layout.reshape(-1), 2))
        if key in seen:
            continue
        seen.add(key)
        proxy = _macro_layout_proxy_cost_np(layout, start_cell_features, pin_features, edge_list)
        displacement = float(np.square(layout.reshape(-1) - current_flat).sum())
        score = proxy + 0.0005 * displacement
        unique.append((score, proxy, displacement, layout.copy()))

    if not unique:
        return [current]
    unique.sort(key=lambda item: item[0])

    # Keep the best proxy candidates plus a few diverse far candidates to avoid
    # collapsing back into one basin.
    selected = [layout for _, _, _, layout in unique[:max_population]]
    if len(unique) > max_population:
        far = sorted(unique[max_population:], key=lambda item: -item[2])[: max(2, max_population // 6)]
        selected.extend(layout for _, _, _, layout in far)

    final = []
    seen = set()
    for layout in selected:
        key = tuple(np.round(layout.reshape(-1), 2))
        if key not in seen:
            seen.add(key)
            final.append(layout)
        if len(final) >= max_population:
            break
    return final




def _mutate_macro_layouts_for_evolution(layouts, start_cell_features, max_mutations=24):
    """Small deterministic mutations for the Lamarckian topology population."""
    if np is None or not layouts:
        return []
    widths = start_cell_features[:, CellFeatureIdx.WIDTH].detach().cpu().numpy().astype(np.float64)
    heights = start_cell_features[:, CellFeatureIdx.HEIGHT].detach().cpu().numpy().astype(np.float64)
    base_positions = start_cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64)
    macros = np.where(heights > 1.5)[0]
    num_macros = int(macros.size)
    if num_macros < 2:
        return []

    def is_legal(layout):
        for left in range(num_macros):
            i = int(macros[left])
            for right in range(left + 1, num_macros):
                j = int(macros[right])
                if (
                    abs(layout[left, 0] - layout[right, 0]) < 0.5 * (widths[i] + widths[j]) - 1e-7
                    and abs(layout[left, 1] - layout[right, 1]) < 0.5 * (heights[i] + heights[j]) - 1e-7
                ):
                    return False
        return True

    mutations = []
    for layout in layouts[: max(1, min(len(layouts), 6))]:
        centroid = layout.mean(axis=0)
        centered = layout - centroid
        for scale_x, scale_y in ((0.96, 1.0), (1.04, 1.0), (1.0, 0.96), (1.0, 1.04), (0.98, 0.98), (1.02, 1.02)):
            mutated = centered.copy()
            mutated[:, 0] *= scale_x
            mutated[:, 1] *= scale_y
            mutated += centroid
            if is_legal(mutated):
                mutations.append(mutated)
        for macro_offset in range(num_macros):
            macro_idx = int(macros[macro_offset])
            step = max(0.5, 0.02 * max(widths[macro_idx], heights[macro_idx]))
            for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                mutated = layout.copy()
                mutated[macro_offset, 0] += dx
                mutated[macro_offset, 1] += dy
                if is_legal(mutated):
                    mutations.append(mutated)
        # A mild recentering toward the original macro centroid sometimes helps
        # when a decoded topology walks too far from useful standard-cell halos.
        original_centroid = base_positions[macros].mean(axis=0)
        shifted = layout + 0.25 * (original_centroid - layout.mean(axis=0))
        if is_legal(shifted):
            mutations.append(shifted)
        if len(mutations) >= max_mutations:
            break
    return mutations[:max_mutations]



