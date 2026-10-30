"""Recursive L-strip boundary augmenter for Kaggle Squares in Squares.

Given a valid seed packing of n_0 squares inside a bounding box of side length s_0,
adds a 1-unit thick L-strip along the top and right edges accommodating
delta_n = 2 * floor(s_0) + 1 additional unit squares in container side s_1 = s_0 + 1 + delta.
Also supports downward subset truncation across intermediate puzzle sizes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.generators.base import BaseGenerator
from src.generators.gobel import GOBEL_CONFIGS, generate_gobel_poses
from src.registry.schema import SquarePose

DEFAULT_LSTRIP_DELTA: float = 2.0e-8


def augment_l_strip(
    seed_poses: List[SquarePose],
    s: float,
    delta: float = DEFAULT_LSTRIP_DELTA,
) -> Tuple[float, List[SquarePose]]:
    """Appends one L-strip layer of axis-aligned unit squares to a seed packing.

    Args:
        seed_poses: Existing valid packing translated to [0, s] x [0, s].
        s: Current bounding box side length of seed_poses.
        delta: Safety clearance margin (>= 1.05e-8).

    Returns:
        Tuple of (new_side_length, augmented_poses).
    """
    k = int(math.floor(s))
    step = 1.0 + delta
    new_s = s + step

    new_poses: List[SquarePose] = [
        SquarePose(cx=p.cx, cy=p.cy, angle=p.angle, square_id=i)
        for i, p in enumerate(seed_poses)
    ]
    sq_id = len(new_poses)

    x_right = s + delta + 0.5
    for j in range(k):
        new_poses.append(
            SquarePose(cx=x_right, cy=0.5 + j * step, angle=0.0, square_id=sq_id)
        )
        sq_id += 1

    y_top = s + delta + 0.5
    for i in range(k):
        new_poses.append(
            SquarePose(cx=0.5 + i * step, cy=y_top, angle=0.0, square_id=sq_id)
        )
        sq_id += 1

    # Corner square
    new_poses.append(
        SquarePose(cx=x_right, cy=y_top, angle=0.0, square_id=sq_id)
    )

    return new_s, new_poses


class LStripAugmenter(BaseGenerator):
    """Generates packings by recursively applying L-strip boundary expansion to Göbel seeds."""

    def __init__(self, delta: float = DEFAULT_LSTRIP_DELTA, max_n: int = 1000):
        self.delta = max(1.05e-8, delta)
        self.max_n = max_n
        self._catalog: Dict[int, Tuple[float, List[SquarePose]]] = {}
        self._build_catalog()

    @property
    def name(self) -> str:
        return "l_strip"

    def _build_catalog(self) -> None:
        """Builds all recursive L-strip augmentations from the 19 Göbel seeds."""
        for seed_n, (a, b) in sorted(GOBEL_CONFIGS.items()):
            s_cur, poses_cur = generate_gobel_poses(a, b, delta=self.delta)
            n_cur = len(poses_cur)
            while True:
                s_next, poses_next = augment_l_strip(poses_cur, s_cur, delta=self.delta)
                n_next = len(poses_next)
                if n_next > self.max_n:
                    break
                grid_ceil = float(math.ceil(math.sqrt(n_next)))
                if s_next < grid_ceil - 1e-6:
                    existing = self._catalog.get(n_next)
                    if existing is None or s_next < existing[0] - 1e-12:
                        self._catalog[n_next] = (s_next, poses_next)
                s_cur, poses_cur, n_cur = s_next, poses_next, n_next

    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Returns the best L-strip augmented packing for puzzle n, or None if not in catalog."""
        entry = self._catalog.get(n)
        if entry is None:
            return None
        _, poses = entry
        return [
            SquarePose(cx=p.cx, cy=p.cy, angle=p.angle, square_id=i)
            for i, p in enumerate(poses)
        ]
