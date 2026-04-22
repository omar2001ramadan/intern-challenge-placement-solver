"""Tests for the experimental branch-and-bound verifier."""

from __future__ import annotations

import unittest

from maxima_proof.branch_and_bound_verifier import (
    ConvexPlacementSubproblem,
    SeparationConstraint,
    tiny_demo_instance,
    verify_global_optimality,
)


class BranchAndBoundVerifierTests(unittest.TestCase):
    def test_constraint_matrix_encodes_left_relation(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        problem = ConvexPlacementSubproblem(cell_features, pin_features, edge_list)
        constraint = problem.constraint_matrix((SeparationConstraint(0, 1, "left"),))

        self.assertEqual(constraint.A.shape, (1, 2))
        self.assertAlmostEqual(float(constraint.A[0, 0]), 1.0)
        self.assertAlmostEqual(float(constraint.lb[0]), 2.0)

    def test_tiny_demo_proves_global_optimality(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        result = verify_global_optimality(
            cell_features,
            pin_features,
            edge_list,
            incumbent_upper=0.707106795,
            incumbent_positions=cell_features[:, 2:4].detach().cpu().numpy(),
            node_limit=20,
            time_limit=10.0,
            tolerance=1e-4,
        )

        self.assertEqual(result.status, "proven")
        self.assertLessEqual(result.gap, 1e-4)


if __name__ == "__main__":
    unittest.main()

