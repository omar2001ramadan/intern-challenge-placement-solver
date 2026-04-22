"""Tests for the sparse LP tangent lower-bound diagnostic."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from placement import calculate_normalized_metrics
from maxima_proof.lp_tangent_certificate import (
    solve_lp_tangent_bound,
    tangent_planes,
    tangent_points,
)
from maxima_proof.lower_bound_certificate import smooth_wirelength_values


class LPTangentCertificateTests(unittest.TestCase):
    def test_tangent_planes_underestimate_random_points(self):
        points = tangent_points(10.0, radial_levels=4)
        gradients, intercepts = tangent_planes(points)
        rng = np.random.default_rng(123)
        for _ in range(100):
            z = rng.normal(size=2) * 5.0
            true_value = float(smooth_wirelength_values(z.reshape(1, 2))[0])
            plane_values = gradients @ z + intercepts
            self.assertLessEqual(float(plane_values.max()), true_value + 1e-9)

    def test_lp_bound_does_not_exceed_legal_manual_solution(self):
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
        lower, *_rest = solve_lp_tangent_bound(cell_features, pin_features, edge_list, radial_levels=4)
        upper = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        self.assertLessEqual(lower, upper + 1e-7)


if __name__ == "__main__":
    unittest.main()

