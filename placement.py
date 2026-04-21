"""Public API wrapper for the VLSI placement challenge submission.

The original challenge imports from `placement.py`, so this file intentionally
keeps that interface stable while the implementation lives in smaller modules.
"""

import torch

from solver.core import (
    MAX_MACRO_AREA,
    MIN_MACRO_AREA,
    MAX_STANDARD_CELL_PINS,
    MIN_STANDARD_CELL_PINS,
    OUTPUT_DIR,
    STANDARD_CELL_AREAS,
    STANDARD_CELL_HEIGHT,
    CellFeatureIdx,
    PinFeatureIdx,
    calculate_cells_with_overlaps,
    calculate_normalized_metrics,
    calculate_overlap_metrics,
    generate_placement_input,
    overlap_repulsion_loss,
    plot_placement,
    wirelength_attraction_loss,
)
from solver.pipeline import train_placement

__all__ = [
    "CellFeatureIdx",
    "PinFeatureIdx",
    "MIN_MACRO_AREA",
    "MAX_MACRO_AREA",
    "STANDARD_CELL_AREAS",
    "STANDARD_CELL_HEIGHT",
    "MIN_STANDARD_CELL_PINS",
    "MAX_STANDARD_CELL_PINS",
    "OUTPUT_DIR",
    "generate_placement_input",
    "wirelength_attraction_loss",
    "overlap_repulsion_loss",
    "train_placement",
    "calculate_overlap_metrics",
    "calculate_cells_with_overlaps",
    "calculate_normalized_metrics",
    "plot_placement",
]


def main():
    """Run a small demonstration placement outside the benchmark harness."""
    torch.manual_seed(42)
    num_macros = 3
    num_std_cells = 50

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros,
        num_std_cells,
    )

    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    result = train_placement(cell_features, pin_features, edge_list, verbose=True, log_interval=200)
    final_cell_features = result["final_cell_features"]
    metrics = calculate_normalized_metrics(final_cell_features, pin_features, edge_list)
    detailed = calculate_overlap_metrics(final_cell_features)

    print(f"Overlap Ratio: {metrics['overlap_ratio']:.4f}")
    print(f"Cells With Overlaps: {metrics['num_cells_with_overlaps']}/{metrics['total_cells']}")
    print(f"Pair Overlap Count: {detailed['overlap_count']}")
    print(f"Normalized Wirelength: {metrics['normalized_wl']:.4f}")

    plot_placement(
        result["initial_cell_features"],
        final_cell_features,
        pin_features,
        edge_list,
        filename="placement_result.png",
    )


if __name__ == "__main__":
    main()
