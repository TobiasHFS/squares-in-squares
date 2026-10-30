"""Corner geometry, bounding boxes, and container side length measurement.

Replicates the exact geometric formulation of Kaggle metric kernel 128385355.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Tuple, List, Optional, Any

try:
    import mpmath
    HAS_MPMATH = True
except ImportError:
    mpmath = None
    HAS_MPMATH = False

# Precision constants matching official metric kernel verbatim
L_DPS = 60
L_DECIMALS = 50


@dataclass(frozen=True, slots=True)
class SquarePose:
    """Immutable public pose of a unit square."""
    cx: float
    cy: float
    angle: float  # radians counter-clockwise


@dataclass(slots=True)
class Square:
    """Internal representation of a unit square with string literals and cached bounds."""
    x: float
    y: float
    angle: float
    x_str: str = ""
    y_str: str = ""
    angle_str: str = ""
    _corners_float: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    _aabb: Tuple[float, float, float, float] | None = field(default=None, repr=False)

    def corners_float(self) -> List[Tuple[float, float]]:
        if not self._corners_float:
            self._corners_float = compute_corners_float(self.x, self.y, self.angle)
        return self._corners_float

    def aabb(self) -> Tuple[float, float, float, float]:
        if self._aabb is None:
            self._aabb = compute_aabb(self.corners_float())
        return self._aabb


def compute_corners_float(cx: float, cy: float, angle: float) -> List[Tuple[float, float]]:
    """Return the 4 corners of a unit square (half-width h=0.5) centered at (cx, cy) rotated by angle.
    
    Order: v0 (top-right), v1 (top-left), v2 (bottom-left), v3 (bottom-right) before rotation.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    h = 0.5
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [
        (cx + dx1, cy + dy1),
        (cx + dx2, cy + dy2),
        (cx - dx1, cy - dy1),
        (cx - dx2, cy - dy2),
    ]


def compute_aabb(corners: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute Axis-Aligned Bounding Box (min_x, min_y, max_x, max_y) from corners."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def aabb_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    """Return True if two AABBs (min_x, min_y, max_x, max_y) intersect in 2D."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def compute_bounding_container_float(corners_list: List[List[Tuple[float, float]]]) -> float:
    """Translation-invariant container side length max(max_x - min_x, max_y - min_y) in float64."""
    if not corners_list:
        return 0.0
    min_x = min(c[0] for corners in corners_list for c in corners)
    max_x = max(c[0] for corners in corners_list for c in corners)
    min_y = min(c[1] for corners in corners_list for c in corners)
    max_y = max(c[1] for corners in corners_list for c in corners)
    return max(max_x - min_x, max_y - min_y)


def compute_corners_mp(cx: Any, cy: Any, angle: Any) -> List[Tuple[Any, Any]]:
    """compute_corners_float at the ambient mpmath.mp precision."""
    if not HAS_MPMATH:
        raise RuntimeError("mpmath is required for arbitrary precision corner calculation")
    c = mpmath.cos(angle)
    s = mpmath.sin(angle)
    h = mpmath.mpf(1) / 2
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [
        (cx + dx1, cy + dy1),
        (cx + dx2, cy + dy2),
        (cx - dx1, cy - dy1),
        (cx - dx2, cy - dy2),
    ]


def compute_corners_interval(cx_iv: Any, cy_iv: Any, angle_iv: Any):
    """compute_corners_float in mpmath.iv interval arithmetic."""
    if not HAS_MPMATH:
        raise RuntimeError("mpmath is required for interval corner calculation")
    c = mpmath.iv.cos(angle_iv)
    s = mpmath.iv.sin(angle_iv)
    h = mpmath.iv.mpf("0.5")
    dx1, dy1 = h * c - h * s, h * s + h * c
    dx2, dy2 = -h * c - h * s, -h * s + h * c
    return [
        (cx_iv + dx1, cy_iv + dy1),
        (cx_iv + dx2, cy_iv + dy2),
        (cx_iv - dx1, cy_iv - dy1),
        (cx_iv - dx2, cy_iv - dy2),
    ]


def measure_side_mp(squares: List[Square], dps: int = L_DPS) -> float:
    """Bounding side max(max_x - min_x, max_y - min_y) at dps digits measured from decimal strings."""
    if HAS_MPMATH:
        old_dps = mpmath.mp.dps
        try:
            mpmath.mp.dps = dps
            xs = []
            ys = []
            for sq in squares:
                corners = compute_corners_mp(
                    mpmath.mpf(sq.x_str if sq.x_str else str(sq.x)),
                    mpmath.mpf(sq.y_str if sq.y_str else str(sq.y)),
                    mpmath.mpf(sq.angle_str if sq.angle_str else str(sq.angle)),
                )
                for x, y in corners:
                    xs.append(x)
                    ys.append(y)
            side = max(max(xs) - min(xs), max(ys) - min(ys))
            return float(side)
        finally:
            mpmath.mp.dps = old_dps
    else:
        # High precision fallback with decimal.Decimal
        getcontext().prec = dps
        xs_dec: List[Decimal] = []
        ys_dec: List[Decimal] = []
        half = Decimal("0.5")
        for sq in squares:
            cx_dec = Decimal(sq.x_str if sq.x_str else f"{sq.x:.17g}")
            cy_dec = Decimal(sq.y_str if sq.y_str else f"{sq.y:.17g}")
            a_val = sq.angle
            cos_a = Decimal(str(math.cos(a_val)))
            sin_a = Decimal(str(math.sin(a_val)))
            dx1 = half * (cos_a - sin_a)
            dy1 = half * (sin_a + cos_a)
            dx2 = -half * (cos_a + sin_a)
            dy2 = half * (-sin_a + cos_a)
            corners_dec = [
                (cx_dec + dx1, cy_dec + dy1),
                (cx_dec + dx2, cy_dec + dy2),
                (cx_dec - dx1, cy_dec - dy1),
                (cx_dec - dx2, cy_dec - dy2),
            ]
            for x_d, y_d in corners_dec:
                xs_dec.append(x_d)
                ys_dec.append(y_d)
        side_dec = max(max(xs_dec) - min(xs_dec), max(ys_dec) - min(ys_dec))
        return float(side_dec)
