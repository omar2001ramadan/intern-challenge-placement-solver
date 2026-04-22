#!/usr/bin/env python3
"""Exact Z3 checker for rationalized MILP lower-bound claims.

SciPy/HiGHS is fast, but it does not emit an independently checkable proof
certificate through SciPy. This module gives the proof folder a second path for
small or capped MILP instances: rebuild the same linear model, convert the
floating coefficients to exact decimal rationals, and ask Z3 whether a solution
exists below a target objective value.

If Z3 returns UNSAT, the claim is independently checked for the rationalized
linear model:

    no modeled placement has objective <= target

This is not a replacement for the full HiGHS branch-and-bound runs on large
instances. It is a proof checker for small/capped instances and a regression
guard for the MILP formulation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

try:  # pragma: no cover - exercised in tests when z3-solver is installed
    import z3
except Exception:  # pragma: no cover - optional dependency
    z3 = None

from placement import generate_placement_input
from test import TEST_CASES

from maxima_proof.milp_branch_verifier import build_milp_model


@dataclass
class Z3CertificateResult:
    status: str
    target_bound: float
    target_raw: str
    num_variables: int
    num_integer_variables: int
    num_constraints: int
    runtime: float
    message: str


def generate_case(num_macros: int, num_std_cells: int, seed: int):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_placement_input(num_macros, num_std_cells)


def _real_value(value: float):
    if not math.isfinite(value):
        raise ValueError("cannot convert non-finite value to Z3")
    # str(float) gives a short decimal that Z3 parses as an exact rational.
    return z3.RealVal(str(float(value)))


def _linear_expr(variable_refs: list, coefficients: np.ndarray):
    terms = []
    for col, value in zip(variable_refs, coefficients):
        if value != 0.0:
            terms.append(_real_value(float(value)) * col)
    if not terms:
        return z3.RealVal(0)
    return z3.Sum(terms)


def prove_no_solution_below(
    cell_features,
    pin_features,
    edge_list,
    *,
    target_bound: float,
    position_bound: float,
    radial_levels: int,
    max_pairs: int | None,
    max_diff_edges: int | None,
    edge_side_tangent_limit: int,
    edge_specific_side_tangents: bool,
    timeout_ms: int,
) -> Z3CertificateResult:
    if z3 is None:
        raise RuntimeError("z3-solver is not installed; run `pip install z3-solver`")

    start = time.time()
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

    objective = model["objective"]
    integrality = model["integrality"]
    bounds = model["bounds"]
    constraints = model["constraints"]
    matrix = constraints.A.tocsr()
    num_vars = len(objective)

    variables = [z3.Real(f"x_{idx}") for idx in range(num_vars)]
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))

    for idx, var in enumerate(variables):
        low = bounds.lb[idx]
        high = bounds.ub[idx]
        if math.isfinite(low):
            solver.add(var >= _real_value(float(low)))
        if math.isfinite(high):
            solver.add(var <= _real_value(float(high)))
        if integrality[idx] == 1:
            solver.add(z3.Or(var == 0, var == 1))

    for row_idx in range(matrix.shape[0]):
        start_ptr = matrix.indptr[row_idx]
        end_ptr = matrix.indptr[row_idx + 1]
        row_cols = matrix.indices[start_ptr:end_ptr]
        row_data = matrix.data[start_ptr:end_ptr]
        expr = _linear_expr([variables[col] for col in row_cols], row_data)
        low = constraints.lb[row_idx]
        high = constraints.ub[row_idx]
        if math.isfinite(low):
            solver.add(expr >= _real_value(float(low)))
        if math.isfinite(high):
            solver.add(expr <= _real_value(float(high)))

    normalizer = max(1, int(edge_list.shape[0])) * math.sqrt(float(cell_features[:, 0].sum().item()))
    target_raw = float(target_bound) * normalizer - float(model["same_total"])
    objective_expr = _linear_expr(variables, objective)
    solver.add(objective_expr <= _real_value(target_raw))

    status = solver.check()
    if status == z3.unsat:
        message = "independent exact UNSAT check succeeded for the rationalized MILP"
        status_text = "unsat"
    elif status == z3.sat:
        message = "Z3 found a modeled solution below the target; the bound is not certified"
        status_text = "sat"
    else:
        message = "Z3 returned unknown before the timeout"
        status_text = "unknown"

    return Z3CertificateResult(
        status=status_text,
        target_bound=float(target_bound),
        target_raw=str(float(target_raw)),
        num_variables=num_vars,
        num_integer_variables=int(integrality.sum()),
        num_constraints=int(matrix.shape[0]),
        runtime=time.time() - start,
        message=message,
    )


def load_case(test_id: int):
    matches = [case for case in TEST_CASES if case[0] == test_id]
    if not matches:
        raise ValueError(f"unknown test case id: {test_id}")
    _, num_macros, num_std_cells, seed = matches[0]
    return generate_case(num_macros, num_std_cells, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int)
    parser.add_argument("--target-bound", type=float, required=True)
    parser.add_argument("--position-bound", type=float, default=500.0)
    parser.add_argument("--radial-levels", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=50, help="set -1 to include every cell pair")
    parser.add_argument("--max-diff-edges", type=int, default=80, help="set -1 to include every different-cell edge")
    parser.add_argument("--edge-side-tangent-limit", type=int, default=0)
    parser.add_argument("--edge-specific-side-tangents", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    return parser.parse_args()


def _none_if_negative(value: int) -> int | None:
    return None if value < 0 else value


def main() -> None:
    args = parse_args()
    if args.case is None:
        raise SystemExit("pass --case <id>")
    cell_features, pin_features, edge_list = load_case(args.case)
    result = prove_no_solution_below(
        cell_features,
        pin_features,
        edge_list,
        target_bound=args.target_bound,
        position_bound=args.position_bound,
        radial_levels=args.radial_levels,
        max_pairs=_none_if_negative(args.max_pairs),
        max_diff_edges=_none_if_negative(args.max_diff_edges),
        edge_side_tangent_limit=args.edge_side_tangent_limit,
        edge_specific_side_tangents=args.edge_specific_side_tangents,
        timeout_ms=args.timeout_ms,
    )
    print("\nZ3 rationalized MILP certificate")
    print("=" * 72)
    print(f"Status: {result.status}")
    print(f"Target normalized bound: {result.target_bound:.9f}")
    print(f"Variables: {result.num_variables}")
    print(f"Binary variables: {result.num_integer_variables}")
    print(f"Constraints: {result.num_constraints}")
    print(f"Runtime: {result.runtime:.2f}s")
    print(f"Message: {result.message}")


if __name__ == "__main__":
    main()
