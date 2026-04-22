"""Tests for the optional exact Z3 MILP checker."""

from __future__ import annotations

import unittest

import torch

try:
    import z3  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    z3 = None

from maxima_proof.z3_milp_certificate import prove_no_solution_below


def tiny_demo_instance() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


@unittest.skipIf(z3 is None, "z3-solver is not installed")
class Z3MILPCertificateTests(unittest.TestCase):
    def test_tiny_demo_unsat_below_known_lower_bound(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        result = prove_no_solution_below(
            cell_features,
            pin_features,
            edge_list,
            target_bound=0.70,
            position_bound=5.0,
            radial_levels=2,
            max_pairs=None,
            max_diff_edges=None,
            edge_side_tangent_limit=4,
            edge_specific_side_tangents=True,
            timeout_ms=5000,
        )

        self.assertEqual(result.status, "unsat")

    def test_tiny_demo_sat_above_known_lower_bound(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        result = prove_no_solution_below(
            cell_features,
            pin_features,
            edge_list,
            target_bound=0.72,
            position_bound=5.0,
            radial_levels=2,
            max_pairs=None,
            max_diff_edges=None,
            edge_side_tangent_limit=4,
            edge_specific_side_tangents=True,
            timeout_ms=5000,
        )

        self.assertEqual(result.status, "sat")


if __name__ == "__main__":
    unittest.main()
