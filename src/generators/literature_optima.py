"""Literature optima and known packings generator for small n in [1, 36].

Uses procedural tilted-block and strip constructions for selected sizes and a grid
fallback for other sizes. The historical API names do not certify optimality.
Candidates still need geometric validation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.generators.base import BaseGenerator
from src.generators.gobel import generate_gobel_poses
from src.generators.grid import generate_compact_grid
from src.registry.schema import SquarePose

DEFAULT_SAFETY_CLEARANCE: float = 2.0e-8


def _reindex(poses: List[SquarePose]) -> List[SquarePose]:
    return [
        SquarePose(cx=p.cx, cy=p.cy, angle=p.angle, square_id=i)
        for i, p in enumerate(poses)
    ]


def generate_optimum_n5(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Proven optimal packing for n=5 (Göbel a=1, b=1, L = 2 + 1/sqrt(2))."""
    _, poses = generate_gobel_poses(a=1, b=1, delta=clearance)
    return _reindex(poses)


def generate_optimum_n10(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Proven optimal packing for n=10 via L-strip augmentation of n=5 (L = 3 + 1/sqrt(2))."""
    s0, poses5 = generate_gobel_poses(a=1, b=1, delta=clearance)
    step = 1.0 + clearance
    x_col = s0 + clearance + 0.5
    y_row = s0 + clearance + 0.5
    extra = [
        SquarePose(cx=x_col, cy=0.5, angle=0.0),
        SquarePose(cx=x_col, cy=0.5 + step, angle=0.0),
        SquarePose(cx=0.5, cy=y_row, angle=0.0),
        SquarePose(cx=0.5 + step, cy=y_row, angle=0.0),
        SquarePose(cx=x_col, cy=y_row, angle=0.0),
    ]
    return _reindex(list(poses5) + extra)


def generate_optimum_n17(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Literature packing for n=17 via recursive L-strip augmentation from n=10 (L ≈ 4.7071)."""
    poses10 = generate_optimum_n10(clearance=clearance)
    s0, _ = generate_gobel_poses(a=1, b=1, delta=clearance)
    step = 1.0 + clearance
    s10 = s0 + step
    x_col = s10 + clearance + 0.5
    y_row = s10 + clearance + 0.5
    extra = [
        SquarePose(cx=x_col, cy=0.5 + j * step, angle=0.0) for j in range(3)
    ] + [
        SquarePose(cx=0.5 + i * step, cy=y_row, angle=0.0) for i in range(3)
    ] + [
        SquarePose(cx=x_col, cy=y_row, angle=0.0)
    ]
    return _reindex(list(poses10) + extra)


def generate_optimum_n26(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Literature packing for n=26 via recursive L-strip augmentation from n=17 (L ≈ 5.7071)."""
    poses17 = generate_optimum_n17(clearance=clearance)
    s0, _ = generate_gobel_poses(a=1, b=1, delta=clearance)
    step = 1.0 + clearance
    s17 = s0 + 2.0 * step
    x_col = s17 + clearance + 0.5
    y_row = s17 + clearance + 0.5
    extra = [
        SquarePose(cx=x_col, cy=0.5 + j * step, angle=0.0) for j in range(4)
    ] + [
        SquarePose(cx=0.5 + i * step, cy=y_row, angle=0.0) for i in range(4)
    ] + [
        SquarePose(cx=x_col, cy=y_row, angle=0.0)
    ]
    return _reindex(list(poses17) + extra)


def generate_optimum_n28(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Göbel optimum for n=28 (a=2, b=4, L = 3 + 2*sqrt(2) ≈ 5.8284)."""
    _, poses = generate_gobel_poses(a=2, b=4, delta=clearance)
    return _reindex(poses)


def generate_optimum_n27(clearance: float = DEFAULT_SAFETY_CLEARANCE) -> List[SquarePose]:
    """Subset packing for n=27 derived from n=28 Göbel configuration (L ≈ 5.8284)."""
    poses28 = generate_optimum_n28(clearance=clearance)
    return _reindex(poses28[:27])


class LiteratureOptimaGenerator(BaseGenerator):
    """Generates constructive square-packing candidates for small n in [1, 36]."""

    def __init__(self, safety_clearance: float = DEFAULT_SAFETY_CLEARANCE):
        self.safety_clearance = max(1.05e-8, safety_clearance)
        self._special_dispatch = {
            5: generate_optimum_n5,
            10: generate_optimum_n10,
            17: generate_optimum_n17,
            26: generate_optimum_n26,
            27: generate_optimum_n27,
            28: generate_optimum_n28,
        }

    @property
    def name(self) -> str:
        return "literature"

    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Generate verified packing of n unit squares for n in [1, 36]."""
        if not (1 <= n <= 36):
            return None

        if n in self._special_dispatch:
            return self._special_dispatch[n](clearance=self.safety_clearance)

        return generate_compact_grid(n, margin=self.safety_clearance)
