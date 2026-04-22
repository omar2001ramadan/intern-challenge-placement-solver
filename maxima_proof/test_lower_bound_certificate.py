"""Smoke and correctness tests for the lower-bound certificate."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from placement import calculate_normalized_metrics
from maxima_proof.lower_bound_certificate import (
    edge_independent_pair_bound,
    generate_case,
    multi_edge_pair_min,
    pairwise_lower_bound,
    run_certificate,
    single_edge_pair_min,
    smooth_wirelength_values,
)


class LowerBoundCertificateTests(unittest.TestCase):
    def test_same_cell_edge_is_counted_exactly(self):
        cell_features = torch.tensor([[4.0, 2.0, 0.0, 0.0, 2.0, 2.0]])
        pin_features = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
                [0.0, 3.0, 4.0, 3.0, 4.0, 0.1, 0.1],
            ]
        )
        edge_list = torch.tensor([[0, 1]], dtype=torch.long)

        lower, same_lower, diff_lower, num_edges, num_pairs = pairwise_lower_bound(
            cell_features,
            pin_features,
            edge_list,
        )
        expected = float(smooth_wirelength_values(np.array([[-3.0, -4.0]]))[0]) / 2.0

        self.assertEqual(num_edges, 1)
        self.assertEqual(num_pairs, 0)
        self.assertAlmostEqual(lower, expected, places=6)
        self.assertAlmostEqual(same_lower, expected, places=6)
        self.assertAlmostEqual(diff_lower, 0.0, places=12)

    def test_single_edge_pair_min_is_below_grid_search(self):
        delta = np.array([0.35, -0.2], dtype=np.float64)
        min_sep_x = 1.25
        min_sep_y = 0.75
        exact_relaxed = single_edge_pair_min(delta, min_sep_x, min_sep_y)

        grid_values = []
        for rx in np.linspace(-2.0, 2.0, 81):
            for ry in np.linspace(-2.0, 2.0, 81):
                if abs(rx) >= min_sep_x or abs(ry) >= min_sep_y:
                    grid_values.append(float(smooth_wirelength_values(np.array([[rx + delta[0], ry + delta[1]]]))[0]))

        self.assertLessEqual(exact_relaxed, min(grid_values) + 1e-9)

    def test_rigorous_pair_bound_is_no_larger_than_bundled_estimate(self):
        deltas = np.array(
            [
                [0.2, -0.3],
                [0.8, 0.1],
                [-0.5, 0.6],
            ],
            dtype=np.float64,
        )
        rigorous = edge_independent_pair_bound(deltas, 1.0, 0.75)
        bundled_estimate = multi_edge_pair_min(deltas, 1.0, 0.75)

        self.assertLessEqual(rigorous, bundled_estimate + 1e-8)

    def test_lower_bound_does_not_exceed_manual_legal_solution(self):
        cell_features = torch.tensor(
            [
                [4.0, 1.0, 0.0, 0.0, 2.0, 2.0],
                [4.0, 1.0, 2.0, 0.0, 2.0, 2.0],
            ]
        )
        pin_features = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
                [1.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.1],
            ]
        )
        edge_list = torch.tensor([[0, 1]], dtype=torch.long)

        lower, *_ = pairwise_lower_bound(cell_features, pin_features, edge_list)
        upper = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]

        self.assertLessEqual(lower, upper + 1e-9)

    def test_rigorous_generated_bound_is_below_known_legal_initial_layout(self):
        cell_features, pin_features, edge_list = generate_case(2, 20, 1001)
        total_cells = cell_features.shape[0]
        spacing = float(cell_features[:, 4:6].max().item()) + 10.0
        for idx in range(total_cells):
            cell_features[idx, 2] = idx * spacing
            cell_features[idx, 3] = 0.0

        lower, *_ = pairwise_lower_bound(cell_features, pin_features, edge_list, mode="rigorous")
        upper = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]

        self.assertEqual(calculate_normalized_metrics(cell_features, pin_features, edge_list)["overlap_ratio"], 0.0)
        self.assertLessEqual(lower, upper + 1e-9)

    def test_certificate_runs_for_first_generated_case(self):
        rows = run_certificate(case_ids=[1], compute_upper_bound=False, mode="rigorous")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].test_id, 1)
        self.assertEqual(rows[0].mode, "rigorous")
        self.assertGreater(rows[0].num_edges, 0)
        self.assertGreaterEqual(rows[0].lower_bound, 0.0)
        self.assertIsNone(rows[0].upper_bound)

    def test_bundled_estimate_runs_but_is_labeled(self):
        rows = run_certificate(case_ids=[1], compute_upper_bound=False, mode="bundled-estimate")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].mode, "bundled-estimate")
        self.assertGreaterEqual(rows[0].lower_bound, 0.0)


if __name__ == "__main__":
    unittest.main()
