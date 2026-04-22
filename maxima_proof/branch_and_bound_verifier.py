#!/usr/bin/env python3
"""Experimental branch-and-bound verifier for placement global optimality.

This is the proof architecture needed for true global optimality:

1. Keep one shared x/y position for every cell.
2. Solve a convex lower-bound subproblem with the overlap choices not yet fixed.
3. If the relaxed solution overlaps, branch on the overlapping pair:
   left, right, above, or below.
4. Prove optimality only when every open branch has a lower bound no better than
   the best legal placement.

The default branch lower bound is exact and conservative: each edge is minimized
independently under the branch constraints that directly apply to its endpoint
pair. This is usually loose, but it is a rigorous lower bound. Numerical convex
node solves remain in this file only for diagnostics and future strengthening;
they are not used by the default proof path.
"""

from __future__ import annotations

import argparse
import contextlib
import heapq
import io
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import torch
from scipy.optimize import LinearConstraint, linprog, minimize

from placement import calculate_normalized_metrics, generate_placement_input, train_placement
from test import TEST_CASES

ALPHA = 0.1
Side = Literal["left", "right", "above", "below"]


@dataclass(frozen=True)
class SeparationConstraint:
    """One chosen side of a pairwise non-overlap disjunction."""

    i: int
    j: int
    side: Side


@dataclass(order=True)
class SearchNode:
    lower_bound: float
    sequence: int
    constraints: tuple[SeparationConstraint, ...] = field(compare=False)
    x0: np.ndarray | None = field(default=None, compare=False)


@dataclass
class VerificationResult:
    status: str
    best_upper_bound: float
    global_lower_bound: float
    gap: float
    nodes_solved: int
    nodes_pruned: int
    nodes_open: int
    runtime: float
    message: str


def _smooth_values_and_grad(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ax = np.abs(z[:, 0]) / ALPHA
    ay = np.abs(z[:, 1]) / ALPHA
    m = np.maximum(ax, ay)
    ex = np.exp(ax - m)
    ey = np.exp(ay - m)
    den = ex + ey
    values = ALPHA * (m + np.log(den))
    grad = np.column_stack(((ex / den) * np.sign(z[:, 0]), (ey / den) * np.sign(z[:, 1])))
    return values, grad


class ConvexPlacementSubproblem:
    """Convex placement objective with a selected set of linear separations."""

    def __init__(self, cell_features: torch.Tensor, pin_features: torch.Tensor, edge_list: torch.Tensor):
        self.cell = cell_features.detach().cpu().numpy().astype(np.float64)
        self.pin = pin_features.detach().cpu().numpy().astype(np.float64)
        self.edges = edge_list.detach().cpu().numpy().astype(np.int64)
        self.num_cells = self.cell.shape[0]
        self.num_vars = max(0, 2 * (self.num_cells - 1))
        self.pin_cell = self.pin[:, 0].astype(np.int64)
        self.pin_offset = self.pin[:, 1:3]
        self.widths = self.cell[:, 4]
        self.heights = self.cell[:, 5]
        self.total_area_sqrt = math.sqrt(float(self.cell[:, 0].sum()))
        self.normalizer = max(1, len(self.edges)) * self.total_area_sqrt
        self.same_cell_constant = self._same_cell_constant()
        self.edge_counts = self._edge_counts()

    def _same_cell_constant(self) -> float:
        total = 0.0
        for u, v in self.edges:
            cu = int(self.pin_cell[u])
            cv = int(self.pin_cell[v])
            if cu == cv:
                values, _ = _smooth_values_and_grad((self.pin_offset[u] - self.pin_offset[v]).reshape(1, 2))
                total += float(values[0])
        return total

    def _edge_counts(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for u, v in self.edges:
            cu = int(self.pin_cell[u])
            cv = int(self.pin_cell[v])
            if cu == cv:
                continue
            pair = (min(cu, cv), max(cu, cv))
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def pack_positions(self, variables: np.ndarray) -> np.ndarray:
        positions = np.zeros((self.num_cells, 2), dtype=np.float64)
        if self.num_cells > 1:
            positions[1:, :] = variables.reshape(self.num_cells - 1, 2)
        return positions

    def unpack_positions(self, positions: np.ndarray) -> np.ndarray:
        if self.num_cells <= 1:
            return np.zeros(0, dtype=np.float64)
        shifted = positions - positions[0]
        return shifted[1:, :].reshape(-1)

    def objective_and_grad(self, variables: np.ndarray) -> tuple[float, np.ndarray]:
        positions = self.pack_positions(variables)
        grad_positions = np.zeros_like(positions)
        total = self.same_cell_constant

        for u, v in self.edges:
            cu = int(self.pin_cell[u])
            cv = int(self.pin_cell[v])
            if cu == cv:
                continue
            z = (
                positions[cu]
                + self.pin_offset[u]
                - positions[cv]
                - self.pin_offset[v]
            ).reshape(1, 2)
            values, grad_z = _smooth_values_and_grad(z)
            total += float(values[0])
            grad_positions[cu] += grad_z[0]
            grad_positions[cv] -= grad_z[0]

        grad_positions -= grad_positions[0]
        grad = grad_positions[1:, :].reshape(-1) if self.num_cells > 1 else np.zeros(0)
        return total / self.normalizer, grad / self.normalizer

    def constraint_matrix(self, constraints: tuple[SeparationConstraint, ...]) -> LinearConstraint | tuple[()]:
        if not constraints or self.num_vars == 0:
            return ()

        rows = []
        lbs = []
        for constraint in constraints:
            row = np.zeros(self.num_vars, dtype=np.float64)

            def add_x(cell_idx: int, coefficient: float) -> None:
                if cell_idx != 0:
                    row[2 * (cell_idx - 1)] += coefficient

            def add_y(cell_idx: int, coefficient: float) -> None:
                if cell_idx != 0:
                    row[2 * (cell_idx - 1) + 1] += coefficient

            sep_x = 0.5 * (self.widths[constraint.i] + self.widths[constraint.j])
            sep_y = 0.5 * (self.heights[constraint.i] + self.heights[constraint.j])

            if constraint.side == "left":
                # i is left of j: x_j - x_i >= sep_x
                add_x(constraint.j, 1.0)
                add_x(constraint.i, -1.0)
                lbs.append(sep_x)
            elif constraint.side == "right":
                add_x(constraint.i, 1.0)
                add_x(constraint.j, -1.0)
                lbs.append(sep_x)
            elif constraint.side == "below":
                add_y(constraint.j, 1.0)
                add_y(constraint.i, -1.0)
                lbs.append(sep_y)
            else:
                add_y(constraint.i, 1.0)
                add_y(constraint.j, -1.0)
                lbs.append(sep_y)

            rows.append(row)

        matrix = np.vstack(rows)
        lower = np.asarray(lbs, dtype=np.float64)
        upper = np.full(len(lbs), np.inf, dtype=np.float64)
        return LinearConstraint(matrix, lower, upper)

    def solve_relaxation(
        self,
        constraints: tuple[SeparationConstraint, ...],
        x0: np.ndarray | None = None,
        maxiter: int = 500,
    ) -> tuple[float, np.ndarray, bool, str]:
        if x0 is None:
            x0 = np.zeros(self.num_vars, dtype=np.float64)
        if constraints:
            linear_constraints = self.constraint_matrix(constraints)
            if np.any(linear_constraints.A @ x0 < linear_constraints.lb - 1e-8):
                feasibility = linprog(
                    np.zeros(self.num_vars, dtype=np.float64),
                    A_ub=-linear_constraints.A,
                    b_ub=-linear_constraints.lb,
                    bounds=[(None, None)] * self.num_vars,
                    method="highs",
                )
                if not feasibility.success:
                    return float("inf"), x0, False, "linear-infeasible"
                x0 = np.asarray(feasibility.x, dtype=np.float64)
            result = minimize(
                lambda x: self.objective_and_grad(x),
                x0,
                jac=True,
                method="SLSQP",
                constraints=linear_constraints,
                options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
            )
        else:
            result = minimize(
                lambda x: self.objective_and_grad(x),
                x0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
            )
        value = float(result.fun) if np.isfinite(result.fun) else float("inf")
        return value, np.asarray(result.x, dtype=np.float64), bool(result.success), str(result.message)

    def largest_overlap(self, positions: np.ndarray) -> tuple[int, int, float] | None:
        best: tuple[int, int, float] | None = None
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                overlap_x = 0.5 * (self.widths[i] + self.widths[j]) - abs(positions[i, 0] - positions[j, 0])
                overlap_y = 0.5 * (self.heights[i] + self.heights[j]) - abs(positions[i, 1] - positions[j, 1])
                if overlap_x > 1e-7 and overlap_y > 1e-7:
                    area = float(overlap_x * overlap_y)
                    if best is None or area > best[2]:
                        best = (i, j, area)
        return best

    def normalized_cost_for_positions(self, positions: np.ndarray) -> float:
        variables = self.unpack_positions(positions)
        value, _ = self.objective_and_grad(variables)
        return value

    def _single_edge_union_min(self, delta: np.ndarray, sep_x: float, sep_y: float) -> float:
        """Exact minimum for one edge under the two-cell non-overlap union."""
        candidates = [
            self._single_edge_halfspace_min(delta, "x_ge", sep_x),
            self._single_edge_halfspace_min(delta, "x_le", -sep_x),
            self._single_edge_halfspace_min(delta, "y_ge", sep_y),
            self._single_edge_halfspace_min(delta, "y_le", -sep_y),
        ]
        return float(min(candidates))

    def _single_edge_halfspace_min(self, delta: np.ndarray, relation: str, threshold: float) -> float:
        """Exact minimum for one edge under one axis-aligned halfspace."""
        rx = -float(delta[0])
        ry = -float(delta[1])
        if relation == "x_ge":
            rx = max(rx, threshold)
        elif relation == "x_le":
            rx = min(rx, threshold)
        elif relation == "y_ge":
            ry = max(ry, threshold)
        elif relation == "y_le":
            ry = min(ry, threshold)
        else:
            raise ValueError(f"unknown halfspace relation: {relation}")
        values, _ = _smooth_values_and_grad(np.array([[rx + delta[0], ry + delta[1]]], dtype=np.float64))
        return float(values[0])

    def _direct_constraint_map(
        self,
        constraints: tuple[SeparationConstraint, ...],
    ) -> dict[tuple[int, int], SeparationConstraint]:
        direct: dict[tuple[int, int], SeparationConstraint] = {}
        for constraint in constraints:
            pair = (min(constraint.i, constraint.j), max(constraint.i, constraint.j))
            direct[pair] = constraint
        return direct

    def _constraint_as_halfspace_for_edge(
        self,
        constraint: SeparationConstraint,
        cell_u: int,
        cell_v: int,
    ) -> tuple[str, float]:
        """Convert a pair side into a halfspace for r = pos_u - pos_v."""
        sep_x = 0.5 * (self.widths[constraint.i] + self.widths[constraint.j])
        sep_y = 0.5 * (self.heights[constraint.i] + self.heights[constraint.j])
        same_orientation = (cell_u == constraint.i and cell_v == constraint.j)

        if constraint.side == "left":
            return ("x_le", -sep_x) if same_orientation else ("x_ge", sep_x)
        if constraint.side == "right":
            return ("x_ge", sep_x) if same_orientation else ("x_le", -sep_x)
        if constraint.side == "below":
            return ("y_le", -sep_y) if same_orientation else ("y_ge", sep_y)
        return ("y_ge", sep_y) if same_orientation else ("y_le", -sep_y)

    def rigorous_node_lower_bound(self, constraints: tuple[SeparationConstraint, ...]) -> float:
        """Exact edge-independent lower bound for a branch node."""
        direct = self._direct_constraint_map(constraints)
        total = self.same_cell_constant

        for u, v in self.edges:
            cu = int(self.pin_cell[u])
            cv = int(self.pin_cell[v])
            if cu == cv:
                continue

            delta = self.pin_offset[u] - self.pin_offset[v]
            pair = (min(cu, cv), max(cu, cv))
            constraint = direct.get(pair)
            if constraint is None:
                sep_x = 0.5 * (self.widths[cu] + self.widths[cv])
                sep_y = 0.5 * (self.heights[cu] + self.heights[cv])
                total += self._single_edge_union_min(delta, sep_x, sep_y)
            else:
                relation, threshold = self._constraint_as_halfspace_for_edge(constraint, cu, cv)
                total += self._single_edge_halfspace_min(delta, relation, threshold)

        return total / self.normalizer

    def all_pairs_constrained(self, constraints: tuple[SeparationConstraint, ...]) -> bool:
        constrained = {(min(c.i, c.j), max(c.i, c.j)) for c in constraints}
        return len(constrained) == self.num_cells * (self.num_cells - 1) // 2

    def choose_unconstrained_pair(self, constraints: tuple[SeparationConstraint, ...]) -> tuple[int, int] | None:
        constrained = {(min(c.i, c.j), max(c.i, c.j)) for c in constraints}
        best_pair = None
        best_score = None
        for i in range(self.num_cells):
            for j in range(i + 1, self.num_cells):
                pair = (i, j)
                if pair in constrained:
                    continue
                sep_area = (self.widths[i] + self.widths[j]) * (self.heights[i] + self.heights[j])
                score = (self.edge_counts.get(pair, 0), sep_area)
                if best_score is None or score > best_score:
                    best_pair = pair
                    best_score = score
        return best_pair

    def legal_features_from_positions(self, positions: np.ndarray) -> torch.Tensor:
        features = torch.tensor(self.cell, dtype=torch.float32)
        features[:, 2:4] = torch.tensor(positions, dtype=torch.float32)
        return features


def generate_case(num_macros: int, num_std_cells: int, seed: int):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_placement_input(num_macros, num_std_cells)


def benchmark_initial_positions(cell_features: torch.Tensor) -> torch.Tensor:
    features = cell_features.clone()
    total_cells = features.shape[0]
    total_area = features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    features[:, 2] = radii * torch.cos(angles)
    features[:, 3] = radii * torch.sin(angles)
    return features


def solver_upper_bound(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
) -> tuple[float, np.ndarray]:
    initialized = benchmark_initial_positions(cell_features)
    result = train_placement(initialized, pin_features, edge_list, verbose=False)
    final_features = result["final_cell_features"]
    metrics = calculate_normalized_metrics(final_features, pin_features, edge_list)
    return float(metrics["normalized_wl"]), final_features[:, 2:4].detach().cpu().numpy().astype(np.float64)


def branch_constraints_for_pair(i: int, j: int) -> tuple[SeparationConstraint, ...]:
    return (
        SeparationConstraint(i, j, "left"),
        SeparationConstraint(i, j, "right"),
        SeparationConstraint(i, j, "below"),
        SeparationConstraint(i, j, "above"),
    )


def verify_global_optimality(
    cell_features: torch.Tensor,
    pin_features: torch.Tensor,
    edge_list: torch.Tensor,
    incumbent_upper: float | None = None,
    incumbent_positions: np.ndarray | None = None,
    node_limit: int = 500,
    time_limit: float = 60.0,
    tolerance: float = 1e-5,
) -> VerificationResult:
    start = time.time()
    problem = ConvexPlacementSubproblem(cell_features, pin_features, edge_list)

    if incumbent_upper is None:
        incumbent_upper, incumbent_positions = solver_upper_bound(cell_features, pin_features, edge_list)
    elif incumbent_positions is None:
        incumbent_positions = np.zeros((problem.num_cells, 2), dtype=np.float64)

    root_value, root_x, root_success, root_message = problem.solve_relaxation(())
    if not root_success:
        root_x = np.zeros(problem.num_vars, dtype=np.float64)
    root_value = problem.rigorous_node_lower_bound(())

    queue: list[SearchNode] = [SearchNode(root_value, 0, (), root_x)]
    sequence = 1
    nodes_solved = 1
    nodes_pruned = 0
    global_lower_bound = root_value

    while queue:
        global_lower_bound = min(node.lower_bound for node in queue)
        gap = incumbent_upper - global_lower_bound
        if gap <= tolerance:
            return VerificationResult(
                status="proven",
                best_upper_bound=incumbent_upper,
                global_lower_bound=global_lower_bound,
                gap=gap,
                nodes_solved=nodes_solved,
                nodes_pruned=nodes_pruned,
                nodes_open=len(queue),
                runtime=time.time() - start,
                message="all remaining branches are bounded by the incumbent",
            )
        if nodes_solved >= node_limit:
            break
        if time.time() - start >= time_limit:
            break

        node = heapq.heappop(queue)
        if node.lower_bound >= incumbent_upper - tolerance:
            nodes_pruned += 1
            continue

        pair = problem.choose_unconstrained_pair(node.constraints)
        if pair is None:
            return VerificationResult(
                status="open",
                best_upper_bound=incumbent_upper,
                global_lower_bound=node.lower_bound,
                gap=incumbent_upper - node.lower_bound,
                nodes_solved=nodes_solved,
                nodes_pruned=nodes_pruned,
                nodes_open=len(queue) + 1,
                runtime=time.time() - start,
                message=(
                    "all pairwise sides are fixed, but the exact convex region "
                    "still needs a certified solver"
                ),
            )

        i, j = pair
        existing = set(node.constraints)
        for branch in branch_constraints_for_pair(i, j):
            if branch in existing:
                continue
            constraints = tuple(sorted(node.constraints + (branch,), key=lambda c: (c.i, c.j, c.side)))
            value = problem.rigorous_node_lower_bound(constraints)
            child_x = node.x0
            success = True
            child_message = "rigorous-edge-independent-bound"
            nodes_solved += 1
            if not success or not np.isfinite(value):
                if "linear-infeasible" in child_message:
                    nodes_pruned += 1
                    continue
                return VerificationResult(
                    status="failed",
                    best_upper_bound=incumbent_upper,
                    global_lower_bound=global_lower_bound,
                    gap=incumbent_upper - global_lower_bound,
                    nodes_solved=nodes_solved,
                    nodes_pruned=nodes_pruned,
                    nodes_open=len(queue),
                    runtime=time.time() - start,
                    message=(
                        "a child convex relaxation failed; numerical failure "
                        "cannot be treated as an infeasibility proof: "
                        f"{child_message}"
                    ),
                )
            if value >= incumbent_upper - tolerance:
                nodes_pruned += 1
                continue
            heapq.heappush(queue, SearchNode(value, sequence, constraints, child_x))
            sequence += 1

    if not queue:
        return VerificationResult(
            status="proven",
            best_upper_bound=incumbent_upper,
            global_lower_bound=incumbent_upper,
            gap=0.0,
            nodes_solved=nodes_solved,
            nodes_pruned=nodes_pruned,
            nodes_open=0,
            runtime=time.time() - start,
            message="all branch regions were pruned or converted into legal incumbents",
        )

    global_lower_bound = min((node.lower_bound for node in queue), default=global_lower_bound)
    return VerificationResult(
        status="open",
        best_upper_bound=incumbent_upper,
        global_lower_bound=global_lower_bound,
        gap=incumbent_upper - global_lower_bound,
        nodes_solved=nodes_solved,
        nodes_pruned=nodes_pruned,
        nodes_open=len(queue),
        runtime=time.time() - start,
        message="node or time budget ended before the branch tree was exhausted",
    )


def tiny_demo_instance() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A two-cell instance small enough for the verifier to close."""
    cell_features = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.0, 2.0, 2.0],
            [4.0, 1.0, 2.0, 0.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    pin_features = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    edge_list = torch.tensor([[0, 1]], dtype=torch.long)
    return cell_features, pin_features, edge_list


def load_test_case(test_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matches = [case for case in TEST_CASES if case[0] == test_id]
    if not matches:
        raise ValueError(f"unknown test case id: {test_id}")
    _, num_macros, num_std_cells, seed = matches[0]
    return generate_case(num_macros, num_std_cells, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the tiny two-cell proof demo")
    parser.add_argument("--case", type=int, help="run one benchmark case")
    parser.add_argument("--node-limit", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        cell_features, pin_features, edge_list = tiny_demo_instance()
        incumbent = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        positions = cell_features[:, 2:4].detach().cpu().numpy().astype(np.float64)
    elif args.case is not None:
        cell_features, pin_features, edge_list = load_test_case(args.case)
        incumbent = None
        positions = None
    else:
        raise SystemExit("pass --demo or --case <id>")

    result = verify_global_optimality(
        cell_features,
        pin_features,
        edge_list,
        incumbent_upper=incumbent,
        incumbent_positions=positions,
        node_limit=args.node_limit,
        time_limit=args.time_limit,
        tolerance=args.tolerance,
    )

    print("\nBranch-and-bound verification result")
    print("=" * 72)
    print(f"Status: {result.status}")
    print(f"Upper bound: {result.best_upper_bound:.9f}")
    print(f"Global lower bound: {result.global_lower_bound:.9f}")
    print(f"Gap: {result.gap:.9f}")
    print(f"Nodes solved: {result.nodes_solved}")
    print(f"Nodes pruned: {result.nodes_pruned}")
    print(f"Open nodes: {result.nodes_open}")
    print(f"Runtime: {result.runtime:.2f}s")
    print(f"Message: {result.message}")


if __name__ == "__main__":
    main()
