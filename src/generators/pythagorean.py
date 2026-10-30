"""Multi-stripe rational Pythagorean angle packing generator.

Uses primitive Pythagorean triples (p, q, r) where p^2 + q^2 = r^2
to construct tilted diagonal stripes with exact rational direction cosines
cos(theta) = q/r, sin(theta) = p/r, surrounded by axis-aligned boundary squares.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.generators.base import BaseGenerator
from src.geometry.corners import Square, SquarePose as GeomSquarePose
from src.geometry.sat import verify_puzzle_collision
from src.registry.schema import SquarePose

DEFAULT_PYTHAGOREAN_DELTA: float = 2.0e-8

PYTHAGOREAN_TRIPLES: Dict[str, Tuple[int, int, int]] = {
    "3-4-5": (3, 4, 5),
    "5-12-13": (5, 12, 13),
    "7-24-25": (7, 24, 25),
    "8-15-17": (8, 15, 17),
}


class PythagoreanGenerator(BaseGenerator):
    """Generates candidate packings with rational Pythagorean tilted stripes."""

    TRIPLES = PYTHAGOREAN_TRIPLES

    def __init__(self, delta: float = DEFAULT_PYTHAGOREAN_DELTA):
        self.delta = max(1.05e-8, delta)

    @property
    def name(self) -> str:
        return "pythagorean"

    def pack_container(
        self,
        L: float,
        p: int,
        q: int,
        num_stripes: int = 1,
    ) -> List[SquarePose]:
        """Packs tilted stripes of angle atan2(p, q) and fills pockets with axis-aligned squares."""
        r = math.hypot(p, q)
        theta = math.atan2(p, q)
        cos_t = q / r
        sin_t = p / r
        step = 1.0 + self.delta
        h_ext = 0.5 * (cos_t + sin_t)

        poses: List[SquarePose] = []
        max_k = int(math.ceil(L * 1.5))

        for s_idx in range(num_stripes):
            v_off = (s_idx - (num_stripes - 1.0) / 2.0) * step
            r0_x = L / 2.0 - v_off * sin_t
            r0_y = L / 2.0 + v_off * cos_t
            for k_idx in range(-max_k, max_k + 1):
                cx = r0_x + k_idx * step * cos_t
                cy = r0_y + k_idx * step * sin_t
                if h_ext <= cx <= L - h_ext and h_ext <= cy <= L - h_ext:
                    poses.append(SquarePose(cx=cx, cy=cy, angle=theta))

        # Fill remaining pockets with axis-aligned grid candidates
        grid_max = int(math.floor(L))
        for row in range(grid_max):
            cy = 0.5 + row * step
            if cy + 0.5 > L:
                continue
            for col in range(grid_max):
                cx = 0.5 + col * step
                if cx + 0.5 > L:
                    continue
                cand = SquarePose(cx=cx, cy=cy, angle=0.0)
                test_poses = poses + [cand]
                sqs = [
                    Square(x=sp.cx, y=sp.cy, angle=sp.angle)
                    for sp in test_poses
                ]
                is_feas, _, pairs = verify_puzzle_collision(
                    sqs, safety_buffer=self.delta
                )
                if is_feas and not pairs:
                    poses.append(cand)

        for idx, sp in enumerate(poses):
            sp.square_id = idx
        return poses

    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Attempts to pack n squares in a container smaller than ceil(sqrt(n))."""
        if not (1 <= n <= 1000):
            return None
        # Only attempt on small/medium non-grid-optimal sizes to keep baseline fast
        if n > 45:
            return None
        k = math.isqrt(n)
        if k * k == n or n == (k + 1) ** 2 - 1:
            return None

        target_L = float(math.ceil(math.sqrt(n))) - 0.02
        if target_L * target_L < n:
            return None

        for p, q, _ in self.TRIPLES.values():
            packed = self.pack_container(target_L, p, q, num_stripes=1)
            if len(packed) >= n:
                trimmed = packed[:n]
                for idx, sp in enumerate(trimmed):
                    sp.square_id = idx
                return trimmed
        return None
