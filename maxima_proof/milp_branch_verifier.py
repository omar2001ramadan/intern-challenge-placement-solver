#!/usr/bin/env python3
"""MILP branch-and-bound lower-bound verifier for placement.

This module builds the branch-and-bound verifier in the most direct form
available from the local Python stack:

* one bounded x/y coordinate is kept for every cell
* each selected cell pair gets four binary side choices
* each selected different-cell edge gets a cost variable
* smooth wirelength is under-estimated by tangent planes
* SciPy's HiGHS MILP backend performs branch-and-bound over the binaries

The result is a lower-bound certificate for the modeled problem. If every edge
and every cell pair is included, it is a bounded-domain certificate for the
tangent relaxation of the full placement problem. If pair or edge caps are used,
the model is a further relaxation: still conservative, but no longer the full
non-overlap MILP.

Important proof scope: big-M non-overlap needs finite coordinate bounds. The
reported proof is therefore over the provided coordinate box. To turn this into
a global optimality proof, one must either prove that the global optimum lies in
that box or use a solver interface with exact indicator constraints plus a
separate boundedness argument.
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import io
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

from placement import calculate_normalized_metrics, generate_placement_input
from test import TEST_CASES

from maxima_proof.lower_bound_certificate import smooth_wirelength_values
from maxima_proof.lp_tangent_certificate import tangent_planes, tangent_points


@dataclass(frozen=True)
class DiffEdge:
    """One different-cell edge in a consistent orientation."""

    src_pin: int
    tgt_pin: int
    src_cell: int
    tgt_cell: int
    offset_delta: np.ndarray


@dataclass
class MILPBranchResult:
    test_id: int
    total_cells: int
    num_edges: int
    total_diff_edges: int
    selected_diff_edges: int
    total_pairs: int
    selected_pairs: int
    tangent_planes: int
    position_bound: float
    lower_bound: float
    solver_primal_bound: float | None
    solver_dual_bound: float | None
    mip_gap: float | None
    node_count: int | None
    solver_success: bool
    solver_status: str
    runtime: float
    used_lp_relaxation_fallback: bool = False
    coordinate_bound_required: float | None = None
    position_bound_certified: bool = False
    cell_graph_connected: bool | None = None
    max_constraint_violation: float | None = None
    max_integrality_violation: float | None = None
    smooth_primal_bound: float | None = None
    tangent_gap_at_primal: float | None = None

    @property
    def is_full_bounded_model(self) -> bool:
        return (
            self.selected_diff_edges == self.total_diff_edges
            and self.selected_pairs == self.total_pairs
        )


@dataclass(frozen=True)
class CoordinateBoundCertificate:
    """Incumbent-derived coordinate box certificate."""

    connected: bool
    required_bound: float | None
    raw_wirelength_upper: float
    reachable_cells: int
    total_cells: int


def generate_case(num_macros: int, num_std_cells: int, seed: int):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_placement_input(num_macros, num_std_cells)


def _same_cell_constant_and_diff_edges(pin_features: torch.Tensor, edge_list: torch.Tensor) -> tuple[float, list[DiffEdge]]:
    pin = pin_features.detach().cpu().numpy().astype(np.float64)
    edges = edge_list.detach().cpu().numpy().astype(np.int64)
    pin_cell = pin[:, 0].astype(np.int64)
    pin_offset = pin[:, 1:3]

    same_total = 0.0
    diff_edges: list[DiffEdge] = []
    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        cell_u = int(pin_cell[u])
        cell_v = int(pin_cell[v])
        delta = pin_offset[u] - pin_offset[v]
        if cell_u == cell_v:
            same_total += float(smooth_wirelength_values(delta.reshape(1, 2))[0])
        else:
            diff_edges.append(DiffEdge(u, v, cell_u, cell_v, delta.astype(np.float64)))
    return same_total, diff_edges


def _select_diff_edges(diff_edges: list[DiffEdge], max_diff_edges: int | None) -> list[DiffEdge]:
    if max_diff_edges is None or max_diff_edges >= len(diff_edges):
        return diff_edges
    ranked = sorted(diff_edges, key=lambda edge: float(np.dot(edge.offset_delta, edge.offset_delta)), reverse=True)
    return ranked[:max_diff_edges]


def _all_cell_pairs(num_cells: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_cells) for j in range(i + 1, num_cells)]


def _select_pairs(
    cell_features: torch.Tensor,
    all_diff_edges: list[DiffEdge],
    selected_diff_edges: list[DiffEdge],
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    num_cells = int(cell_features.shape[0])
    pairs = _all_cell_pairs(num_cells)
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs

    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    widths = cell[:, 4]
    heights = cell[:, 5]
    edge_count: dict[tuple[int, int], int] = {}
    selected_edge_count: dict[tuple[int, int], int] = {}

    for edge in all_diff_edges:
        pair = (min(edge.src_cell, edge.tgt_cell), max(edge.src_cell, edge.tgt_cell))
        edge_count[pair] = edge_count.get(pair, 0) + 1
    for edge in selected_diff_edges:
        pair = (min(edge.src_cell, edge.tgt_cell), max(edge.src_cell, edge.tgt_cell))
        selected_edge_count[pair] = selected_edge_count.get(pair, 0) + 1

    def pair_score(pair: tuple[int, int]) -> tuple[float, float, float]:
        i, j = pair
        sep_area = (widths[i] + widths[j]) * (heights[i] + heights[j])
        size_signal = max(widths[i], heights[i]) + max(widths[j], heights[j])
        return (
            float(selected_edge_count.get(pair, 0)),
            float(edge_count.get(pair, 0)),
            float(sep_area + 0.05 * size_signal),
        )

    return sorted(pairs, key=pair_score, reverse=True)[:max_pairs]


def _edge_side_tangent_points(
    cell_features: torch.Tensor,
    diff_edges: list[DiffEdge],
    edge_side_tangent_limit: int,
) -> np.ndarray:
    if edge_side_tangent_limit <= 0 or not diff_edges:
        return np.zeros((0, 2), dtype=np.float64)

    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    widths = cell[:, 4]
    heights = cell[:, 5]
    points: list[tuple[float, float]] = []
    ranked = sorted(diff_edges, key=lambda edge: float(np.dot(edge.offset_delta, edge.offset_delta)), reverse=True)

    for edge in ranked[:edge_side_tangent_limit]:
        delta = edge.offset_delta
        sep_x = 0.5 * float(widths[edge.src_cell] + widths[edge.tgt_cell])
        sep_y = 0.5 * float(heights[edge.src_cell] + heights[edge.tgt_cell])

        candidates = [
            (max(sep_x, -delta[0]), -delta[1]),
            (min(-sep_x, -delta[0]), -delta[1]),
            (-delta[0], max(sep_y, -delta[1])),
            (-delta[0], min(-sep_y, -delta[1])),
        ]
        for rx, ry in candidates:
            points.append((float(rx + delta[0]), float(ry + delta[1])))

    unique: list[tuple[float, float]] = []
    seen = set()
    for point in points:
        key = (round(point[0], 8), round(point[1], 8))
        if key in seen:
            continue
        seen.add(key)
        unique.append(point)
    return np.asarray(unique, dtype=np.float64)


def _single_edge_side_tangent_points(cell_features: torch.Tensor, edge: DiffEdge) -> np.ndarray:
    """Return side-minimizer tangent points for one edge only."""
    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    widths = cell[:, 4]
    heights = cell[:, 5]
    delta = edge.offset_delta
    sep_x = 0.5 * float(widths[edge.src_cell] + widths[edge.tgt_cell])
    sep_y = 0.5 * float(heights[edge.src_cell] + heights[edge.tgt_cell])
    return np.asarray(
        [
            (max(sep_x, -delta[0]) + delta[0], -delta[1] + delta[1]),
            (min(-sep_x, -delta[0]) + delta[0], -delta[1] + delta[1]),
            (-delta[0] + delta[0], max(sep_y, -delta[1]) + delta[1]),
            (-delta[0] + delta[0], min(-sep_y, -delta[1]) + delta[1]),
        ],
        dtype=np.float64,
    )


def build_tangent_library(
    cell_features: torch.Tensor,
    selected_diff_edges: list[DiffEdge],
    radial_levels: int,
    edge_side_tangent_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    max_dim = float(cell[:, 4:6].max(initial=1.0))
    base_points = tangent_points(max(4.0 * max_dim, 10.0), radial_levels=radial_levels)
    side_points = _edge_side_tangent_points(cell_features, selected_diff_edges, edge_side_tangent_limit)
    if side_points.size:
        points = np.vstack([base_points, side_points])
    else:
        points = base_points

    unique_points = []
    seen = set()
    for point in points:
        key = (round(float(point[0]), 8), round(float(point[1]), 8))
        if key in seen:
            continue
        seen.add(key)
        unique_points.append((float(point[0]), float(point[1])))
    return tangent_planes(np.asarray(unique_points, dtype=np.float64))


def _position_col(cell_idx: int, axis: int) -> int:
    return 2 * cell_idx + axis


def _binary_col(num_pos: int, num_t: int, pair_index: int, side_offset: int) -> int:
    return num_pos + num_t + 4 * pair_index + side_offset


def derive_coordinate_bound_certificate(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    normalized_upper_bound: float,
) -> CoordinateBoundCertificate:
    """Prove a coordinate box from an incumbent upper bound when possible.

    If the different-cell connectivity graph is connected, any placement whose
    normalized objective is no worse than `normalized_upper_bound` can be
    translated so cell 0 is at the origin and every cell lies inside the returned
    box. The argument is conservative: the total raw upper bound limits every
    individual edge cost because all edge costs are nonnegative.
    """
    same_total, diff_edges = _same_cell_constant_and_diff_edges(pin_features, edge_list)
    num_cells = int(cell_features.shape[0])
    total_edges = int(edge_list.shape[0])
    total_area_sqrt = math.sqrt(float(cell_features[:, 0].sum().item()))
    raw_upper = float(normalized_upper_bound) * max(1, total_edges) * total_area_sqrt

    graph: list[list[tuple[int, float]]] = [[] for _ in range(num_cells)]
    for edge in diff_edges:
        weight = max(
            raw_upper + abs(float(edge.offset_delta[0])),
            raw_upper + abs(float(edge.offset_delta[1])),
        )
        graph[edge.src_cell].append((edge.tgt_cell, weight))
        graph[edge.tgt_cell].append((edge.src_cell, weight))

    distances = [float("inf")] * num_cells
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, cell_idx = heapq.heappop(queue)
        if distance != distances[cell_idx]:
            continue
        for neighbor, weight in graph[cell_idx]:
            next_distance = distance + weight
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                heapq.heappush(queue, (next_distance, neighbor))

    finite_distances = [value for value in distances if math.isfinite(value)]
    connected = len(finite_distances) == num_cells
    required = max(finite_distances) if connected else None
    return CoordinateBoundCertificate(
        connected=connected,
        required_bound=required,
        raw_wirelength_upper=raw_upper,
        reachable_cells=len(finite_distances),
        total_cells=num_cells,
    )


def build_milp_model(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    position_bound: float,
    radial_levels: int,
    max_pairs: int | None,
    max_diff_edges: int | None,
    edge_side_tangent_limit: int,
    edge_specific_side_tangents: bool = False,
):
    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    widths = cell[:, 4]
    heights = cell[:, 5]
    num_cells = int(cell.shape[0])
    total_pairs = num_cells * (num_cells - 1) // 2
    same_total, all_diff_edges = _same_cell_constant_and_diff_edges(pin_features, edge_list)
    selected_diff_edges = _select_diff_edges(all_diff_edges, max_diff_edges)
    selected_pairs = _select_pairs(cell_features, all_diff_edges, selected_diff_edges, max_pairs)
    pair_to_index = {pair: idx for idx, pair in enumerate(selected_pairs)}

    gradients, intercepts = build_tangent_library(
        cell_features,
        selected_diff_edges,
        radial_levels=radial_levels,
        edge_side_tangent_limit=edge_side_tangent_limit,
    )
    num_planes = int(gradients.shape[0])

    num_pos = 2 * num_cells
    num_t = len(selected_diff_edges)
    num_binary = 4 * len(selected_pairs)
    num_vars = num_pos + num_t + num_binary

    objective = np.zeros(num_vars, dtype=np.float64)
    if num_t:
        objective[num_pos : num_pos + num_t] = 1.0

    lower_bounds = np.full(num_vars, -np.inf, dtype=np.float64)
    upper_bounds = np.full(num_vars, np.inf, dtype=np.float64)
    lower_bounds[:num_pos] = -float(position_bound)
    upper_bounds[:num_pos] = float(position_bound)
    # Translation anchor. Without this, all legal placements can be shifted.
    lower_bounds[_position_col(0, 0)] = 0.0
    upper_bounds[_position_col(0, 0)] = 0.0
    lower_bounds[_position_col(0, 1)] = 0.0
    upper_bounds[_position_col(0, 1)] = 0.0
    if num_t:
        lower_bounds[num_pos : num_pos + num_t] = 0.0
    if num_binary:
        lower_bounds[num_pos + num_t :] = 0.0
        upper_bounds[num_pos + num_t :] = 1.0

    integrality = np.zeros(num_vars, dtype=np.int8)
    if num_binary:
        integrality[num_pos + num_t :] = 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []
    row_idx = 0

    def add(row: int, col: int, value: float) -> None:
        if value == 0.0:
            return
        rows.append(row)
        cols.append(col)
        data.append(float(value))

    for edge_idx, edge in enumerate(selected_diff_edges):
        t_col = num_pos + edge_idx
        for grad, intercept in zip(gradients, intercepts):
            add(row_idx, t_col, 1.0)
            add(row_idx, _position_col(edge.src_cell, 0), -float(grad[0]))
            add(row_idx, _position_col(edge.src_cell, 1), -float(grad[1]))
            add(row_idx, _position_col(edge.tgt_cell, 0), float(grad[0]))
            add(row_idx, _position_col(edge.tgt_cell, 1), float(grad[1]))
            lbs.append(float(np.dot(grad, edge.offset_delta) + intercept))
            ubs.append(np.inf)
            row_idx += 1
        if edge_specific_side_tangents:
            edge_gradients, edge_intercepts = tangent_planes(_single_edge_side_tangent_points(cell_features, edge))
            for grad, intercept in zip(edge_gradients, edge_intercepts):
                add(row_idx, t_col, 1.0)
                add(row_idx, _position_col(edge.src_cell, 0), -float(grad[0]))
                add(row_idx, _position_col(edge.src_cell, 1), -float(grad[1]))
                add(row_idx, _position_col(edge.tgt_cell, 0), float(grad[0]))
                add(row_idx, _position_col(edge.tgt_cell, 1), float(grad[1]))
                lbs.append(float(np.dot(grad, edge.offset_delta) + intercept))
                ubs.append(np.inf)
                row_idx += 1

    for pair_index, (i, j) in enumerate(selected_pairs):
        sep_x = 0.5 * float(widths[i] + widths[j])
        sep_y = 0.5 * float(heights[i] + heights[j])
        big_m = 2.0 * float(position_bound) + max(sep_x, sep_y) + 1.0
        left = _binary_col(num_pos, num_t, pair_index, 0)
        right = _binary_col(num_pos, num_t, pair_index, 1)
        below = _binary_col(num_pos, num_t, pair_index, 2)
        above = _binary_col(num_pos, num_t, pair_index, 3)

        # i left of j: x_j - x_i >= sep_x when the left binary is active.
        add(row_idx, _position_col(j, 0), 1.0)
        add(row_idx, _position_col(i, 0), -1.0)
        add(row_idx, left, -big_m)
        lbs.append(sep_x - big_m)
        ubs.append(np.inf)
        row_idx += 1

        # i right of j: x_i - x_j >= sep_x.
        add(row_idx, _position_col(i, 0), 1.0)
        add(row_idx, _position_col(j, 0), -1.0)
        add(row_idx, right, -big_m)
        lbs.append(sep_x - big_m)
        ubs.append(np.inf)
        row_idx += 1

        # i below j: y_j - y_i >= sep_y.
        add(row_idx, _position_col(j, 1), 1.0)
        add(row_idx, _position_col(i, 1), -1.0)
        add(row_idx, below, -big_m)
        lbs.append(sep_y - big_m)
        ubs.append(np.inf)
        row_idx += 1

        # i above j: y_i - y_j >= sep_y.
        add(row_idx, _position_col(i, 1), 1.0)
        add(row_idx, _position_col(j, 1), -1.0)
        add(row_idx, above, -big_m)
        lbs.append(sep_y - big_m)
        ubs.append(np.inf)
        row_idx += 1

        # Exactly one disjunct is chosen. Any legal pair can select one true side.
        add(row_idx, left, 1.0)
        add(row_idx, right, 1.0)
        add(row_idx, below, 1.0)
        add(row_idx, above, 1.0)
        lbs.append(1.0)
        ubs.append(1.0)
        row_idx += 1

    constraints = LinearConstraint(
        coo_matrix((data, (rows, cols)), shape=(row_idx, num_vars)).tocsr(),
        np.asarray(lbs, dtype=np.float64),
        np.asarray(ubs, dtype=np.float64),
    )
    bounds = Bounds(lower_bounds, upper_bounds)

    return {
        "same_total": same_total,
        "all_diff_edges": all_diff_edges,
        "selected_diff_edges": selected_diff_edges,
        "total_pairs": total_pairs,
        "selected_pairs": selected_pairs,
        "num_planes": num_planes,
        "objective": objective,
        "integrality": integrality,
        "bounds": bounds,
        "constraints": constraints,
    }


def _smooth_selected_wirelength_from_solution(model: dict, solution: np.ndarray, cell_features: torch.Tensor) -> float:
    num_cells = int(cell_features.shape[0])
    num_pos = 2 * num_cells
    positions = solution[:num_pos].reshape(num_cells, 2)
    total = float(model["same_total"])
    for edge in model["selected_diff_edges"]:
        delta = positions[edge.src_cell] - positions[edge.tgt_cell] + edge.offset_delta
        total += float(smooth_wirelength_values(delta.reshape(1, 2))[0])
    normalizer = max(1, int(model["num_total_edges"])) * math.sqrt(float(cell_features[:, 0].sum().item()))
    return total / normalizer


def _audit_solution(model: dict, solution: np.ndarray, integrality: np.ndarray) -> tuple[float, float]:
    linear_constraint: LinearConstraint = model["constraints"]
    activity = linear_constraint.A @ solution
    lower_violation = np.maximum(linear_constraint.lb - activity, 0.0)
    upper_violation = np.maximum(activity - linear_constraint.ub, 0.0)
    constraint_violation = float(max(lower_violation.max(initial=0.0), upper_violation.max(initial=0.0)))
    integer_values = solution[integrality == 1]
    if integer_values.size == 0:
        integrality_violation = 0.0
    else:
        integrality_violation = float(np.max(np.abs(integer_values - np.round(integer_values))))
    return constraint_violation, integrality_violation


def _bounds_as_pairs(bounds: Bounds) -> list[tuple[float | None, float | None]]:
    pairs = []
    for low, high in zip(bounds.lb, bounds.ub):
        low_value = None if not np.isfinite(low) else float(low)
        high_value = None if not np.isfinite(high) else float(high)
        pairs.append((low_value, high_value))
    return pairs


def _solve_lp_relaxation_dual_bound(model: dict, time_limit: float) -> tuple[float, str]:
    """Solve the continuous relaxation as a fallback lower bound.

    A MIP time limit can expire before HiGHS reports a useful MIP dual bound.
    The LP relaxation is weaker than branch-and-bound, but it is still a valid
    lower bound for the same modeled problem.
    """
    linear_constraint: LinearConstraint = model["constraints"]
    matrix = linear_constraint.A
    lower = linear_constraint.lb
    upper = linear_constraint.ub

    a_ub_parts = []
    b_ub_parts = []

    finite_upper = np.isfinite(upper)
    if finite_upper.any():
        a_ub_parts.append(matrix[finite_upper])
        b_ub_parts.append(upper[finite_upper])

    finite_lower = np.isfinite(lower)
    if finite_lower.any():
        a_ub_parts.append(-matrix[finite_lower])
        b_ub_parts.append(-lower[finite_lower])

    if a_ub_parts:
        a_ub = vstack(a_ub_parts).tocsr()
        b_ub = np.concatenate(b_ub_parts).astype(np.float64)
    else:
        a_ub = None
        b_ub = None

    result = linprog(
        model["objective"],
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=_bounds_as_pairs(model["bounds"]),
        method="highs",
        options={"time_limit": max(1.0, float(time_limit)), "presolve": True},
    )
    if result.success and result.fun is not None and np.isfinite(result.fun):
        return float(result.fun), f"LP relaxation fallback: {result.message}"
    return float("nan"), f"LP relaxation fallback failed: {result.message}"


def solve_milp_branch_bound(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    *,
    test_id: int = 0,
    position_bound: float = 500.0,
    radial_levels: int = 3,
    max_pairs: int | None = None,
    max_diff_edges: int | None = None,
    edge_side_tangent_limit: int = 0,
    edge_specific_side_tangents: bool = False,
    coordinate_upper_bound: float | None = None,
    time_limit: float = 60.0,
    mip_rel_gap: float = 1e-4,
) -> MILPBranchResult:
    start = time.time()
    coordinate_certificate = (
        derive_coordinate_bound_certificate(cell_features, pin_features, edge_list, coordinate_upper_bound)
        if coordinate_upper_bound is not None
        else None
    )
    model = build_milp_model(
        cell_features,
        pin_features,
        edge_list,
        position_bound=position_bound,
        radial_levels=radial_levels,
        max_pairs=max_pairs,
        max_diff_edges=max_diff_edges,
        edge_side_tangent_limit=edge_side_tangent_limit,
        edge_specific_side_tangents=edge_specific_side_tangents,
    )
    model["num_total_edges"] = int(edge_list.shape[0])

    result = milp(
        c=model["objective"],
        integrality=model["integrality"],
        bounds=model["bounds"],
        constraints=model["constraints"],
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": True,
            "disp": False,
        },
    )

    total_edges = int(edge_list.shape[0])
    normalizer = max(1, total_edges) * math.sqrt(float(cell_features[:, 0].sum().item()))
    same_total = float(model["same_total"])

    raw_dual = getattr(result, "mip_dual_bound", None)
    used_lp_relaxation_fallback = False
    fallback_status = ""
    if raw_dual is None or not np.isfinite(raw_dual):
        if bool(result.success) and result.fun is not None and np.isfinite(result.fun):
            raw_dual = float(result.fun)
        else:
            raw_dual, fallback_status = _solve_lp_relaxation_dual_bound(model, time_limit=min(30.0, time_limit))
            used_lp_relaxation_fallback = np.isfinite(raw_dual)
    raw_primal = float(result.fun) if result.fun is not None and np.isfinite(result.fun) else None

    lower_bound = (same_total + float(raw_dual)) / normalizer if np.isfinite(raw_dual) else float("nan")
    primal_bound = (same_total + raw_primal) / normalizer if raw_primal is not None else None
    solver_status = str(result.message)
    if fallback_status:
        solver_status = f"{solver_status}; {fallback_status}"
    max_constraint_violation = None
    max_integrality_violation = None
    smooth_primal_bound = None
    tangent_gap_at_primal = None
    if result.x is not None:
        max_constraint_violation, max_integrality_violation = _audit_solution(model, np.asarray(result.x), model["integrality"])
        if raw_primal is not None:
            smooth_primal_bound = _smooth_selected_wirelength_from_solution(model, np.asarray(result.x), cell_features)
            tangent_gap_at_primal = smooth_primal_bound - primal_bound if primal_bound is not None else None

    certified_bound = (
        coordinate_certificate.required_bound is not None
        and coordinate_certificate.required_bound <= float(position_bound)
    ) if coordinate_certificate is not None else False

    return MILPBranchResult(
        test_id=test_id,
        total_cells=int(cell_features.shape[0]),
        num_edges=total_edges,
        total_diff_edges=len(model["all_diff_edges"]),
        selected_diff_edges=len(model["selected_diff_edges"]),
        total_pairs=int(model["total_pairs"]),
        selected_pairs=len(model["selected_pairs"]),
        tangent_planes=int(model["num_planes"]),
        position_bound=float(position_bound),
        lower_bound=lower_bound,
        solver_primal_bound=primal_bound,
        solver_dual_bound=(same_total + float(raw_dual)) / normalizer if np.isfinite(raw_dual) else None,
        mip_gap=float(getattr(result, "mip_gap", float("nan"))) if getattr(result, "mip_gap", None) is not None else None,
        node_count=int(getattr(result, "mip_node_count", 0)) if getattr(result, "mip_node_count", None) is not None else None,
        solver_success=bool(result.success),
        solver_status=solver_status,
        runtime=time.time() - start,
        used_lp_relaxation_fallback=used_lp_relaxation_fallback,
        coordinate_bound_required=coordinate_certificate.required_bound if coordinate_certificate is not None else None,
        position_bound_certified=bool(certified_bound),
        cell_graph_connected=coordinate_certificate.connected if coordinate_certificate is not None else None,
        max_constraint_violation=max_constraint_violation,
        max_integrality_violation=max_integrality_violation,
        smooth_primal_bound=smooth_primal_bound,
        tangent_gap_at_primal=tangent_gap_at_primal,
    )


def run_milp_branch_certificate(
    case_ids: Iterable[int] | None = None,
    *,
    position_bound: float = 500.0,
    radial_levels: int = 3,
    max_pairs: int | None = 600,
    max_diff_edges: int | None = 3000,
    edge_side_tangent_limit: int = 0,
    edge_specific_side_tangents: bool = False,
    coordinate_upper_bound: float | None = None,
    time_limit: float = 60.0,
    mip_rel_gap: float = 1e-4,
) -> list[MILPBranchResult]:
    wanted = set(case_ids) if case_ids is not None else None
    rows: list[MILPBranchResult] = []
    for test_id, num_macros, num_std_cells, seed in TEST_CASES:
        if wanted is not None and test_id not in wanted:
            continue
        cell, pin, edges = generate_case(num_macros, num_std_cells, seed)
        rows.append(
            solve_milp_branch_bound(
                cell,
                pin,
                edges,
                test_id=test_id,
                position_bound=position_bound,
                radial_levels=radial_levels,
                max_pairs=max_pairs,
                max_diff_edges=max_diff_edges,
                edge_side_tangent_limit=edge_side_tangent_limit,
                edge_specific_side_tangents=edge_specific_side_tangents,
                coordinate_upper_bound=coordinate_upper_bound,
                time_limit=time_limit,
                mip_rel_gap=mip_rel_gap,
            )
        )
    return rows


def print_results(rows: list[MILPBranchResult]) -> None:
    print("\nMILP branch-and-bound tangent verifier")
    print("=" * 126)
    print(
        f"{'Test':>4} {'Cells':>6} {'Edges':>7} {'DiffE':>11} {'Pairs':>11} "
        f"{'Planes':>6} {'LB':>10} {'Primal':>10} {'Gap':>9} {'Nodes':>7} {'Time':>8} {'Full':>5} {'LPfb':>5} {'Box':>5}"
    )
    for row in rows:
        primal = "n/a" if row.solver_primal_bound is None else f"{row.solver_primal_bound:.6f}"
        gap = "n/a" if row.mip_gap is None or not math.isfinite(row.mip_gap) else f"{row.mip_gap:.2e}"
        nodes = "n/a" if row.node_count is None else str(row.node_count)
        print(
            f"{row.test_id:>4} {row.total_cells:>6} {row.num_edges:>7} "
            f"{row.selected_diff_edges:>5}/{row.total_diff_edges:<5} "
            f"{row.selected_pairs:>5}/{row.total_pairs:<5} "
            f"{row.tangent_planes:>6} {row.lower_bound:>10.6f} "
            f"{primal:>10} {gap:>9} {nodes:>7} {row.runtime:>8.2f} "
            f"{str(row.is_full_bounded_model):>5} {str(row.used_lp_relaxation_fallback):>5} {str(row.position_bound_certified):>5}"
        )
    finite = [row for row in rows if math.isfinite(row.lower_bound)]
    if finite:
        print("-" * 126)
        print(f"Average MILP lower bound: {sum(row.lower_bound for row in finite) / len(finite):.6f}")
    print("\nInterpretation:")
    print("  HiGHS solves the binary side-choice model with branch-and-bound.")
    print("  The reported LB uses the MILP dual bound, so time-limited runs remain conservative.")
    print("  Full=True means every pair and every different-cell edge was included.")
    print("  LPfb=True means the MILP timed out before a dual bound and used its LP relaxation.")
    print("  Box=True means the position bound was certified from the supplied upper bound.")
    print("  The certificate is bounded by the stated coordinate box used for big-M.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=int)
    parser.add_argument("--position-bound", type=float, default=500.0)
    parser.add_argument("--radial-levels", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=600, help="set -1 to include every cell pair")
    parser.add_argument("--max-diff-edges", type=int, default=3000, help="set -1 to include every different-cell edge")
    parser.add_argument("--edge-side-tangent-limit", type=int, default=0)
    parser.add_argument("--edge-specific-side-tangents", action="store_true")
    parser.add_argument("--coordinate-upper-bound", type=float)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    return parser.parse_args()


def _none_if_negative(value: int) -> int | None:
    return None if value < 0 else value


def main() -> None:
    args = parse_args()
    rows = run_milp_branch_certificate(
        args.cases,
        position_bound=args.position_bound,
        radial_levels=args.radial_levels,
        max_pairs=_none_if_negative(args.max_pairs),
        max_diff_edges=_none_if_negative(args.max_diff_edges),
        edge_side_tangent_limit=args.edge_side_tangent_limit,
        edge_specific_side_tangents=args.edge_specific_side_tangents,
        coordinate_upper_bound=args.coordinate_upper_bound,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap,
    )
    print_results(rows)


if __name__ == "__main__":
    main()
