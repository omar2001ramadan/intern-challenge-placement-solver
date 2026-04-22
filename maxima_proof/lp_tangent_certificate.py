#!/usr/bin/env python3
"""Sparse LP tangent-relaxation lower-bound diagnostic.

This module builds a stronger global lower-bound relaxation than the exact
edge-independent certificate:

* one shared x/y position is kept for every cell
* every different-cell edge gets a cost variable
* the smooth wirelength cost is under-estimated by tangent planes
* the resulting linear program is solved with SciPy/HiGHS

Mathematically, tangent planes of a convex function are valid global lower
bounds. The LP therefore models a valid relaxation if solved exactly. In this
implementation, the LP is solved in floating point, so the result is reported as
a residual-audited diagnostic lower bound rather than a machine-checkable proof.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from placement import calculate_normalized_metrics, generate_placement_input
from test import TEST_CASES

from maxima_proof.lower_bound_certificate import smooth_wirelength_values

ALPHA = 0.1


@dataclass
class LPTangentResult:
    test_id: int
    total_cells: int
    num_edges: int
    diff_edges: int
    tangent_planes: int
    lower_bound: float
    same_cell_part: float
    lp_part: float
    solver_success: bool
    solver_status: str
    max_constraint_violation: float
    runtime: float


def generate_case(num_macros: int, num_std_cells: int, seed: int):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_placement_input(num_macros, num_std_cells)


def tangent_points(max_scale: float, radial_levels: int = 6) -> np.ndarray:
    """Create points where tangent planes are sampled."""
    directions = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]
    radii = np.geomspace(0.05, max(max_scale, 0.1), num=radial_levels)
    points = [(0.0, 0.0)]
    for radius in radii:
        for dx, dy in directions:
            norm = math.hypot(dx, dy)
            points.append((radius * dx / norm, radius * dy / norm))
    return np.asarray(points, dtype=np.float64)


def tangent_planes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent gradients and intercepts for smooth wirelength."""
    gradients = []
    intercepts = []
    for point in points:
        z = point.reshape(1, 2)
        ax = abs(point[0]) / ALPHA
        ay = abs(point[1]) / ALPHA
        m = max(ax, ay)
        ex = math.exp(ax - m)
        ey = math.exp(ay - m)
        den = ex + ey
        value = float(smooth_wirelength_values(z)[0])
        grad = np.array(
            [
                (ex / den) * math.copysign(1.0, point[0]) if point[0] != 0.0 else 0.0,
                (ey / den) * math.copysign(1.0, point[1]) if point[1] != 0.0 else 0.0,
            ],
            dtype=np.float64,
        )
        gradients.append(grad)
        intercepts.append(value - float(np.dot(grad, point)))
    return np.asarray(gradients), np.asarray(intercepts)


def same_cell_constant_and_diff_edges(cell_features, pin_features, edge_list):
    pin = pin_features.detach().cpu().numpy().astype(np.float64)
    edges = edge_list.detach().cpu().numpy().astype(np.int64)
    pin_cell = pin[:, 0].astype(np.int64)
    pin_offset = pin[:, 1:3]

    same_total = 0.0
    diff = []
    for u, v in edges:
        cu = int(pin_cell[u])
        cv = int(pin_cell[v])
        if cu == cv:
            same_total += float(smooth_wirelength_values((pin_offset[u] - pin_offset[v]).reshape(1, 2))[0])
        else:
            diff.append((int(u), int(v), cu, cv, pin_offset[u] - pin_offset[v]))
    return same_total, diff


def select_diff_edges(diff_edges, max_diff_edges: int | None):
    """Choose a deterministic subset of nonnegative edge terms for a partial LP."""
    if max_diff_edges is None or max_diff_edges >= len(diff_edges):
        return diff_edges
    # Dropping edge terms is conservative because smooth wirelength is nonnegative.
    # Keep edges with larger pin-offset norms because they tend to carry stronger
    # lower-bound signal.
    ranked = sorted(diff_edges, key=lambda item: float(np.dot(item[4], item[4])), reverse=True)
    return ranked[:max_diff_edges]


def position_column(cell_idx: int, axis: int) -> int | None:
    """Column for a free position variable before positive/negative splitting."""
    if cell_idx == 0:
        return None
    return 2 * (cell_idx - 1) + axis


def build_lp_matrices(cell_features, pin_features, edge_list, radial_levels: int, max_diff_edges: int | None = None):
    cell = cell_features.detach().cpu().numpy().astype(np.float64)
    num_cells = cell.shape[0]
    same_total, all_diff_edges = same_cell_constant_and_diff_edges(cell_features, pin_features, edge_list)
    diff_edges = select_diff_edges(all_diff_edges, max_diff_edges)
    if not diff_edges:
        return same_total, diff_edges, None, None, None, 0

    max_dim = float(cell[:, 4:6].max())
    gradients, intercepts = tangent_planes(tangent_points(max(4.0 * max_dim, 10.0), radial_levels=radial_levels))
    num_planes = gradients.shape[0]

    num_pos = max(0, 2 * (num_cells - 1))
    num_split_pos = 2 * num_pos
    num_t = len(diff_edges)
    num_vars = num_split_pos + num_t

    rows = []
    cols = []
    data = []
    rhs = []
    row_idx = 0

    for edge_idx, (u, v, cu, cv, offset_delta) in enumerate(diff_edges):
        t_col = num_split_pos + edge_idx
        for grad, intercept in zip(gradients, intercepts):
            # t_e - grad dot (pos_cu - pos_cv) >= grad dot offset_delta + intercept.
            rows.append(row_idx)
            cols.append(t_col)
            data.append(1.0)

            for cell_idx, sign in ((cu, -1.0), (cv, 1.0)):
                for axis in (0, 1):
                    base_col = position_column(cell_idx, axis)
                    if base_col is None:
                        continue
                    coeff = sign * float(grad[axis])
                    # p = p_plus - p_minus.
                    rows.append(row_idx)
                    cols.append(base_col)
                    data.append(coeff)
                    rows.append(row_idx)
                    cols.append(num_pos + base_col)
                    data.append(-coeff)

            rhs.append(float(np.dot(grad, offset_delta) + intercept))
            row_idx += 1

    # linprog uses A_ub x <= b_ub, so negate Gx >= h.
    G = coo_matrix((data, (rows, cols)), shape=(row_idx, num_vars)).tocsr()
    A_ub = -G
    b_ub = -np.asarray(rhs, dtype=np.float64)

    objective = np.zeros(num_vars, dtype=np.float64)
    objective[num_split_pos:] = 1.0
    bounds = [(0.0, None)] * num_vars
    return same_total, diff_edges, objective, A_ub, b_ub, num_planes


def solve_lp_tangent_bound(
    cell_features,
    pin_features,
    edge_list,
    radial_levels: int = 5,
    max_diff_edges: int | None = None,
) -> tuple[float, float, float, bool, str, float, int, int]:
    start = time.time()
    same_total, diff_edges, objective, A_ub, b_ub, num_planes = build_lp_matrices(
        cell_features,
        pin_features,
        edge_list,
        radial_levels=radial_levels,
        max_diff_edges=max_diff_edges,
    )
    total_edges = int(edge_list.shape[0])
    normalizer = max(1, total_edges) * math.sqrt(float(cell_features[:, 0].sum().item()))

    if objective is None:
        lower = same_total / normalizer
        return lower, same_total / normalizer, 0.0, True, "no different-cell edges", 0.0, 0, 0

    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * len(objective),
        method="highs",
        options={"presolve": True, "dual_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8},
    )
    if result.x is None:
        lower = float("nan")
        max_violation = float("nan")
        lp_part = float("nan")
    else:
        residual = A_ub @ result.x - b_ub
        max_violation = float(max(0.0, residual.max(initial=0.0)))
        lp_part = float(result.fun) / normalizer
        lower = (same_total + float(result.fun)) / normalizer

    return (
        lower,
        same_total / normalizer,
        lp_part,
        bool(result.success),
        str(result.message),
        max_violation,
        len(diff_edges),
        num_planes,
    )


def run_lp_tangent_certificate(
    case_ids: Iterable[int] | None = None,
    radial_levels: int = 5,
    max_diff_edges: int | None = None,
) -> list[LPTangentResult]:
    wanted = set(case_ids) if case_ids is not None else None
    rows = []
    for test_id, num_macros, num_std_cells, seed in TEST_CASES:
        if wanted is not None and test_id not in wanted:
            continue
        cell, pin, edges = generate_case(num_macros, num_std_cells, seed)
        start = time.time()
        lower, same_part, lp_part, success, status, violation, diff_edges, num_planes = solve_lp_tangent_bound(
            cell,
            pin,
            edges,
            radial_levels=radial_levels,
            max_diff_edges=max_diff_edges,
        )
        rows.append(
            LPTangentResult(
                test_id=test_id,
                total_cells=num_macros + num_std_cells,
                num_edges=int(edges.shape[0]),
                diff_edges=diff_edges,
                tangent_planes=num_planes,
                lower_bound=lower,
                same_cell_part=same_part,
                lp_part=lp_part,
                solver_success=success,
                solver_status=status,
                max_constraint_violation=violation,
                runtime=time.time() - start,
            )
        )
    return rows


def print_results(rows: list[LPTangentResult]) -> None:
    print("\nSparse LP tangent-relaxation lower-bound diagnostic")
    print("=" * 104)
    print(f"{'Test':>4} {'Cells':>6} {'Edges':>7} {'DiffE':>7} {'Planes':>6} {'LP LB':>10} {'Viol':>10} {'Time':>8} {'OK':>4}")
    for row in rows:
        print(
            f"{row.test_id:>4} {row.total_cells:>6} {row.num_edges:>7} {row.diff_edges:>7} "
            f"{row.tangent_planes:>6} {row.lower_bound:>10.6f} "
            f"{row.max_constraint_violation:>10.2e} {row.runtime:>8.2f} {str(row.solver_success):>4}"
        )
    successful = [row for row in rows if row.solver_success and math.isfinite(row.lower_bound)]
    if successful:
        print("-" * 104)
        print(f"Average successful LP lower bound: {sum(row.lower_bound for row in successful) / len(successful):.6f}")
    print("\nInterpretation:")
    print("  Tangent planes are valid lower bounds of the convex wirelength term.")
    print("  The LP keeps global x/y consistency, so it is often stronger than edge-independent bounds.")
    print("  Because SciPy/HiGHS is floating point, this is a residual-audited diagnostic, not exact arithmetic proof.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", type=int)
    parser.add_argument("--radial-levels", type=int, default=5)
    parser.add_argument(
        "--max-diff-edges",
        type=int,
        help="optional deterministic subset size for large cases; omitted edges are nonnegative and safely dropped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_lp_tangent_certificate(
        args.cases,
        radial_levels=args.radial_levels,
        max_diff_edges=args.max_diff_edges,
    )
    print_results(rows)


if __name__ == "__main__":
    main()
