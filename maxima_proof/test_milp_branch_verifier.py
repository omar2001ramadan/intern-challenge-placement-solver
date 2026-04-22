"""Tests for the MILP branch-and-bound tangent verifier."""

from __future__ import annotations

import unittest

import torch

from placement import calculate_normalized_metrics
from maxima_proof.milp_branch_verifier import (
    build_milp_model,
    derive_coordinate_bound_certificate,
    solve_milp_branch_bound,
)


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


class MILPBranchVerifierTests(unittest.TestCase):
    def test_model_contains_four_binary_side_choices_for_one_pair(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        model = build_milp_model(
            cell_features,
            pin_features,
            edge_list,
            position_bound=5.0,
            radial_levels=2,
            max_pairs=None,
            max_diff_edges=None,
            edge_side_tangent_limit=4,
        )

        self.assertEqual(len(model["selected_pairs"]), 1)
        self.assertEqual(int(model["integrality"].sum()), 4)
        self.assertEqual(model["constraints"].A.shape[0], model["num_planes"] + 5)

    def test_tiny_demo_milp_lower_bound_matches_legal_solution(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        upper = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        result = solve_milp_branch_bound(
            cell_features,
            pin_features,
            edge_list,
            position_bound=5.0,
            radial_levels=2,
            max_pairs=None,
            max_diff_edges=None,
            edge_side_tangent_limit=4,
            edge_specific_side_tangents=True,
            coordinate_upper_bound=upper,
            time_limit=10.0,
            mip_rel_gap=1e-9,
        )

        self.assertTrue(result.solver_success)
        self.assertTrue(result.is_full_bounded_model)
        self.assertTrue(result.position_bound_certified)
        self.assertLessEqual(result.lower_bound, upper + 1e-7)
        self.assertAlmostEqual(result.lower_bound, upper, places=6)

    def test_coordinate_bound_certificate_for_connected_tiny_case(self):
        cell_features, pin_features, edge_list = tiny_demo_instance()
        upper = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        certificate = derive_coordinate_bound_certificate(cell_features, pin_features, edge_list, upper)

        self.assertTrue(certificate.connected)
        self.assertEqual(certificate.reachable_cells, 2)
        self.assertAlmostEqual(certificate.required_bound, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
