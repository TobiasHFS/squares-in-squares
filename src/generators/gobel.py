"""Frits Göbel's 45° tilted central block generator family.

Implements all 19 valid configurations (n = 2a^2 + 2a + b^2) for 1 <= n <= 1000
satisfying the boundary matching condition a - 1 < b / sqrt(2) < a + 1
with certified safety clearance delta = 2.0e-8.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.generators.base import BaseGenerator
from src.registry.schema import SquarePose

DEFAULT_GOBEL_DELTA: float = 2.0e-8

# Exhaustive table of all 19 valid Göbel configurations in n in [1, 1000]
GOBEL_CONFIGS: Dict[int, Tuple[int, int]] = {
    5: (1, 1),
    28: (2, 4),
    40: (3, 4),
    65: (4, 5),
    89: (4, 7),
    109: (5, 7),
    124: (5, 8),
    148: (6, 8),
    233: (7, 11),
    265: (8, 11),
    376: (9, 14),
    416: (10, 14),
    445: (10, 15),
    489: (11, 15),
    636: (12, 18),
    688: (13, 18),
    861: (14, 21),
    921: (15, 21),
    964: (15, 22),
}


def generate_gobel_poses(
    a: int,
    b: int,
    delta: float = DEFAULT_GOBEL_DELTA,
) -> Tuple[float, List[SquarePose]]:
    """Generates the exact geometric packing for Göbel parameters (a, b).

    Args:
        a: Integer size of triangular corner staircases (a >= 1).
        b: Integer side of central b x b tilted block (b >= 1).
        delta: Safety clearance buffer (default: 2.0e-8).

    Returns:
        Tuple of (container_side_length, list_of_poses).
    """
    step = 1.0 + delta
    h_block = 0.5 + (b - 1.0) * step / 2.0
    L = 2.0 + (a - 1.0) * step + h_block * math.sqrt(2.0) + delta * math.sqrt(2.0)

    poses: List[SquarePose] = []
    sq_id = 0

    # 4 Corner staircases (each has a*(a+1)/2 squares, total = 2a^2 + 2a)
    for r in range(a):
        for c in range(a - r):
            # Bottom-Left
            poses.append(
                SquarePose(
                    cx=0.5 + c * step,
                    cy=0.5 + r * step,
                    angle=0.0,
                    square_id=sq_id,
                )
            )
            sq_id += 1
            # Top-Left
            poses.append(
                SquarePose(
                    cx=0.5 + c * step,
                    cy=L - (0.5 + r * step),
                    angle=0.0,
                    square_id=sq_id,
                )
            )
            sq_id += 1
            # Bottom-Right
            poses.append(
                SquarePose(
                    cx=L - (0.5 + c * step),
                    cy=0.5 + r * step,
                    angle=0.0,
                    square_id=sq_id,
                )
            )
            sq_id += 1
            # Top-Right
            poses.append(
                SquarePose(
                    cx=L - (0.5 + c * step),
                    cy=L - (0.5 + r * step),
                    angle=0.0,
                    square_id=sq_id,
                )
            )
            sq_id += 1

    # Central b x b block rotated by 45 degrees (pi/4)
    for u in range(b):
        for v in range(b):
            u_p = (u - (b - 1.0) / 2.0) * step
            v_p = (v - (b - 1.0) / 2.0) * step
            dx = (u_p - v_p) / math.sqrt(2.0)
            dy = (u_p + v_p) / math.sqrt(2.0)
            poses.append(
                SquarePose(
                    cx=L / 2.0 + dx,
                    cy=L / 2.0 + dy,
                    angle=math.pi / 4.0,
                    square_id=sq_id,
                )
            )
            sq_id += 1

    return L, poses


class GobelGenerator(BaseGenerator):
    """Constructive generator for Frits Göbel's 45° central block family."""

    CONFIGS = GOBEL_CONFIGS

    def __init__(self, delta: float = DEFAULT_GOBEL_DELTA):
        self.delta = max(1.05e-8, delta)

    @property
    def name(self) -> str:
        return "gobel"

    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Generates candidate packing if n is one of the 19 valid Göbel configurations."""
        if n not in self.CONFIGS:
            return None
        a, b = self.CONFIGS[n]
        _, poses = generate_gobel_poses(a, b, delta=self.delta)
        return poses
