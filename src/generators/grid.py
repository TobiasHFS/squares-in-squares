"""Exact rigid compact grid generator for Kaggle Squares in Squares.

Generates compact axis-aligned grids with safety clearance margin margin >= 1.05e-8
ensuring 0 overlapping pairs under Separating Axis Theorem (SAT) validation.
For rigid regimes (n = k^2 and n = k^2 - 1), achieves exact optimal side length L <= k + 1e-6.
"""

from __future__ import annotations

import math
from typing import List, Optional

from src.generators.base import BaseGenerator
from src.registry.schema import SquarePose

DEFAULT_GRID_MARGIN: float = 1.1e-8


def generate_compact_grid(n: int, margin: float = DEFAULT_GRID_MARGIN) -> List[SquarePose]:
    """Generates an axis-aligned compact grid packing of n unit squares.

    Args:
        n: Puzzle size (1 <= n <= 1000).
        margin: Safety clearance buffer between adjacent squares (default: 1.1e-8).

    Returns:
        List of n SquarePose objects positioned on the compact grid.
    """
    if n <= 0:
        return []

    k = math.isqrt(n - 1) + 1 if n > 1 else 1
    step = 1.0 + margin
    poses: List[SquarePose] = []

    for i in range(n):
        col = i % k
        row = i // k
        poses.append(
            SquarePose(
                cx=0.5 + col * step,
                cy=0.5 + row * step,
                angle=0.0,
                square_id=i,
            )
        )

    return poses


class GridGenerator(BaseGenerator):
    """Generates compact axis-aligned grids for any puzzle size n in [1, 1000]."""

    def __init__(self, margin: float = DEFAULT_GRID_MARGIN):
        self.margin = max(1.05e-8, margin)

    @property
    def name(self) -> str:
        return "grid"

    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Generates compact grid packing for puzzle size n."""
        if not (1 <= n <= 1000):
            return None
        return generate_compact_grid(n, margin=self.margin)
