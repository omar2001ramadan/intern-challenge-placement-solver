#!/usr/bin/env python3
"""Lower-bound certificate for the placement challenge.

This module does not prove that the submitted placement is globally optimal. It
proves a weaker but valid statement:

    no legal placement can have normalized wirelength below the reported lower
    bound from this relaxation.

The default certificate is intentionally conservative: every different-cell
wire is minimized independently subject to the exact two-cell non-overlap rule.
That drops global consistency, and it also drops consistency between multiple
wires connecting the same cell pair. Dropping constraints can only make a
minimization problem easier, so the resulting value is a valid lower bound.

The module also contains a tighter bundled pair estimate. It groups all wires
between the same unordered cell pair and solves a small convex problem with
SciPy. That estimate is useful diagnostics, but it is not the default formal
certificate because a numerical optimizer is not a proof by itself.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import minimize

from placement import (
    calculate_normalized_metrics,
    generate_placement_input,
    train_placement,
)


ALPHA = 0.1

TEST_CASES = [
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
]


@dataclass
class CertificateResult:
    test_id: int
    total_cells: int
    num_macros: int
    num_std_cells: int
    seed: int
    num_edges: int
    num_cell_pairs: int
    lower_bound: float
    same_cell_lower_bound: float
    different_cell_lower_bound: float
    mode: str
    upper_bound: float | None = None
    upper_overlap: float | None = None
    upper_runtime: float | None = None

    @property
    def gap(self) -> float | None:
        if self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound


def smooth_wirelength_values(z: np.ndarray) -> np.ndarray:
    """Challenge smooth wirelength primitive for rows of x and y deltas."""
    ax = np.abs(z[:, 0]) / ALPHA
    ay = np.abs(z[:, 1]) / ALPHA
    m = np.maximum(ax, ay)
    return ALPHA * (m + np.log(np.exp(ax - m) + np.exp(ay - m)))


def bundle_objective_and_grad(r: np.ndarray, deltas: np.ndarray) -> tuple[float, np.ndarray]:
    """Sum smooth wirelength values for one cell pair and return a gradient."""
    z = deltas + r[None, :]
    ax = np.abs(z[:, 0]) / ALPHA
    ay = np.abs(z[:, 1]) / ALPHA
    m = np.maximum(ax, ay)
    ex = np.exp(ax - m)
    ey = np.exp(ay - m)
    den = ex + ey

    value = ALPHA * float(np.sum(m + np.log(den)))
    grad = np.array(
        [
            np.sum((ex / den) * np.sign(z[:, 0])),
            np.sum((ey / den) * np.sign(z[:, 1])),
        ],
        dtype=np.float64,
    )
    return value, grad


def single_edge_pair_min(delta: np.ndarray, min_sep_x: float, min_sep_y: float) -> float:
    """Independent lower bound for one edge between two different cells."""
    candidates = []

    rx = max(min_sep_x, -delta[0])
    ry = -delta[1]
    candidates.append(smooth_wirelength_values(np.array([[rx + delta[0], ry + delta[1]]]))[0])

    rx = min(-min_sep_x, -delta[0])
    ry = -delta[1]
    candidates.append(smooth_wirelength_values(np.array([[rx + delta[0], ry + delta[1]]]))[0])

    rx = -delta[0]
    ry = max(min_sep_y, -delta[1])
    candidates.append(smooth_wirelength_values(np.array([[rx + delta[0], ry + delta[1]]]))[0])

    rx = -delta[0]
    ry = min(-min_sep_y, -delta[1])
    candidates.append(smooth_wirelength_values(np.array([[rx + delta[0], ry + delta[1]]]))[0])

    return float(min(candidates))


def edge_independent_pair_bound(deltas: np.ndarray, min_sep_x: float, min_sep_y: float) -> float:
    """Rigorous lower bound that lets each edge choose its own separation."""
    return float(sum(single_edge_pair_min(delta, min_sep_x, min_sep_y) for delta in deltas))


def multi_edge_pair_min(deltas: np.ndarray, min_sep_x: float, min_sep_y: float) -> float:
    """Numerical bundled estimate for all edges between one unordered cell pair.

    This is mathematically a valid relaxation if solved exactly, but this
    implementation uses a numerical optimizer. Use it for diagnostics, not as
    the default formal certificate.
    """
    if len(deltas) == 1:
        return single_edge_pair_min(deltas[0], min_sep_x, min_sep_y)

    center = -np.median(deltas, axis=0)
    best = float("inf")
    side_specs = [
        ([(min_sep_x, None), (None, None)], np.array([max(center[0], min_sep_x), center[1]])),
        ([(None, -min_sep_x), (None, None)], np.array([min(center[0], -min_sep_x), center[1]])),
        ([(None, None), (min_sep_y, None)], np.array([center[0], max(center[1], min_sep_y)])),
        ([(None, None), (None, -min_sep_y)], np.array([center[0], min(center[1], -min_sep_y)])),
    ]

    for bounds, x0 in side_specs:
        result = minimize(
            lambda r: bundle_objective_and_grad(r, deltas),
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-10, "gtol": 1e-8, "maxls": 30},
        )
        value = float(result.fun)
        if not np.isfinite(value):
            value = bundle_objective_and_grad(result.x, deltas)[0]
        best = min(best, value)

    return best


def generate_case(num_macros: int, num_std_cells: int, seed: int):
    """Generate one deterministic challenge case without printing generator logs."""
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_placement_input(num_macros, num_std_cells)


def initialize_like_benchmark(cell_features: torch.Tensor) -> torch.Tensor:
    """Apply the same initial position spread used by `test.py`.

    The benchmark sets the seed before generating the netlist, then uses the
    continued Torch random state for initial positions. Do not reset the seed
    here, or the upper-bound path will no longer match `test.py`.
    """
    features = cell_features.clone()
    total_cells = features.shape[0]
    total_area = features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    features[:, 2] = radii * torch.cos(angles)
    features[:, 3] = radii * torch.sin(angles)
    return features


def pairwise_lower_bound(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    mode: str = "rigorous",
) -> tuple[float, float, float, int, int]:
    """Compute the normalized lower bound.

    Modes:
      - rigorous: exact edge-independent lower bound. This is the proof mode.
      - bundled-estimate: tighter numerical pair-bundled relaxation estimate.
    """
    if mode not in {"rigorous", "bundled-estimate"}:
        raise ValueError(f"unknown certificate mode: {mode}")

    if edge_list.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0, 0

    cell = cell_features.detach().cpu().numpy()
    pin = pin_features.detach().cpu().numpy()
    edges = edge_list.detach().cpu().numpy()

    pin_cell = pin[:, 0].astype(np.int64)
    pin_offset = pin[:, 1:3]
    widths = cell[:, 4]
    heights = cell[:, 5]

    same_cell_total = 0.0
    grouped_deltas: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for raw_u, raw_v in edges:
        u = int(raw_u)
        v = int(raw_v)
        cell_u = int(pin_cell[u])
        cell_v = int(pin_cell[v])

        if cell_u == cell_v:
            same_cell_total += float(smooth_wirelength_values((pin_offset[u] - pin_offset[v]).reshape(1, 2))[0])
            continue

        left, right = (cell_u, cell_v) if cell_u < cell_v else (cell_v, cell_u)
        if cell_u == left:
            delta = pin_offset[u] - pin_offset[v]
        else:
            delta = pin_offset[v] - pin_offset[u]
        grouped_deltas[(left, right)].append(delta)

    different_cell_total = 0.0
    for (left, right), deltas in grouped_deltas.items():
        min_sep_x = 0.5 * float(widths[left] + widths[right])
        min_sep_y = 0.5 * float(heights[left] + heights[right])
        deltas_array = np.asarray(deltas, dtype=np.float64)
        if mode == "rigorous":
            different_cell_total += edge_independent_pair_bound(
                deltas_array,
                min_sep_x,
                min_sep_y,
            )
        else:
            different_cell_total += multi_edge_pair_min(
                deltas_array,
                min_sep_x,
                min_sep_y,
            )

    total = same_cell_total + different_cell_total
    normalizer = len(edges) * math.sqrt(float(cell[:, 0].sum()))
    return (
        total / normalizer,
        same_cell_total / normalizer,
        different_cell_total / normalizer,
        len(edges),
        len(grouped_deltas),
    )


def compute_current_upper_bound(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
) -> tuple[float, float, float]:
    """Run the actual solver and return normalized wirelength, overlap, runtime."""
    initialized = initialize_like_benchmark(cell_features)
    start = time.time()
    result = train_placement(initialized, pin_features, edge_list, verbose=False)
    runtime = time.time() - start
    metrics = calculate_normalized_metrics(result["final_cell_features"], pin_features, edge_list)
    return metrics["normalized_wl"], metrics["overlap_ratio"], runtime


def selected_cases(case_ids: Iterable[int] | None = None) -> list[tuple[int, int, int, int]]:
    if case_ids is None:
        return TEST_CASES
    wanted = set(case_ids)
    cases = [case for case in TEST_CASES if case[0] in wanted]
    missing = wanted.difference(case[0] for case in cases)
    if missing:
        raise ValueError(f"unknown test case ids: {sorted(missing)}")
    return cases


def run_certificate(
    case_ids: Iterable[int] | None = None,
    compute_upper_bound: bool = False,
    mode: str = "rigorous",
) -> list[CertificateResult]:
    """Run the certificate for selected benchmark cases."""
    rows = []

    for test_id, num_macros, num_std_cells, seed in selected_cases(case_ids):
        cell_features, pin_features, edge_list = generate_case(num_macros, num_std_cells, seed)
        lower, same_lower, diff_lower, num_edges, num_pairs = pairwise_lower_bound(
            cell_features,
            pin_features,
            edge_list,
            mode=mode,
        )

        upper = None
        overlap = None
        upper_runtime = None
        if compute_upper_bound:
            upper, overlap, upper_runtime = compute_current_upper_bound(
                cell_features,
                pin_features,
                edge_list,
            )

        rows.append(
            CertificateResult(
                test_id=test_id,
                total_cells=num_macros + num_std_cells,
                num_macros=num_macros,
                num_std_cells=num_std_cells,
                seed=seed,
                num_edges=num_edges,
                num_cell_pairs=num_pairs,
                lower_bound=lower,
                same_cell_lower_bound=same_lower,
                different_cell_lower_bound=diff_lower,
                mode=mode,
                upper_bound=upper,
                upper_overlap=overlap,
                upper_runtime=upper_runtime,
            )
        )

    return rows


def print_table(rows: list[CertificateResult], runtime: float) -> None:
    has_upper = any(row.upper_bound is not None for row in rows)
    mode = rows[0].mode if rows else "rigorous"
    title = "Rigorous edge-independent lower-bound certificate"
    if mode == "bundled-estimate":
        title = "Numerical bundled pair-relaxation estimate"
    print(f"\n{title}")
    print("=" * 88)
    if has_upper:
        print(f"{'Test':>4} {'Cells':>6} {'Edges':>7} {'Pairs':>7} {'LB':>10} {'UB':>10} {'Gap':>10} {'Overlap':>9}")
    else:
        print(f"{'Test':>4} {'Cells':>6} {'Edges':>7} {'Pairs':>7} {'LB':>10} {'Same LB':>10} {'Diff LB':>10}")

    for row in rows:
        if has_upper:
            upper = "n/a" if row.upper_bound is None else f"{row.upper_bound:.6f}"
            gap = "n/a" if row.gap is None else f"{row.gap:.6f}"
            overlap = "n/a" if row.upper_overlap is None else f"{row.upper_overlap:.6f}"
            print(
                f"{row.test_id:>4} {row.total_cells:>6} {row.num_edges:>7} "
                f"{row.num_cell_pairs:>7} {row.lower_bound:>10.6f} "
                f"{upper:>10} {gap:>10} {overlap:>9}"
            )
        else:
            print(
                f"{row.test_id:>4} {row.total_cells:>6} {row.num_edges:>7} "
                f"{row.num_cell_pairs:>7} {row.lower_bound:>10.6f} "
                f"{row.same_cell_lower_bound:>10.6f} {row.different_cell_lower_bound:>10.6f}"
            )

    average_lower = sum(row.lower_bound for row in rows) / len(rows)
    print("-" * 88)
    print(f"Average lower bound: {average_lower:.6f}")
    if has_upper:
        upper_rows = [row for row in rows if row.upper_bound is not None]
        average_upper = sum(row.upper_bound for row in upper_rows if row.upper_bound is not None) / len(upper_rows)
        print(f"Average upper bound: {average_upper:.6f}")
        print(f"Average certificate gap: {average_upper - average_lower:.6f}")
    print(f"Runtime: {runtime:.2f}s")
    print("\nInterpretation:")
    if mode == "rigorous":
        print("  The lower bound is a valid mathematical floor from an exact relaxation.")
    else:
        print("  This is a tighter numerical relaxation estimate, not a formal proof.")
    print("  The upper bound, when computed, is the current legal solver result.")
    print("  A small gap is strong evidence; a large gap means the proof is weak.")


def write_json(rows: list[CertificateResult], path: Path) -> None:
    payload = []
    for row in rows:
        item = asdict(row)
        item["gap"] = row.gap
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        type=int,
        help="test case ids to run; defaults to all first ten cases",
    )
    parser.add_argument(
        "--compute-upper-bound",
        action="store_true",
        help="also run the placement solver to compute current legal upper bounds",
    )
    parser.add_argument(
        "--mode",
        choices=("rigorous", "bundled-estimate"),
        default="rigorous",
        help="certificate mode; default is the rigorous edge-independent proof",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional path for machine-readable certificate results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    rows = run_certificate(args.cases, compute_upper_bound=args.compute_upper_bound, mode=args.mode)
    runtime = time.time() - start
    print_table(rows, runtime)
    if args.json_out is not None:
        write_json(rows, args.json_out)


if __name__ == "__main__":
    main()
