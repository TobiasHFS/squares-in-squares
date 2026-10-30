"""Separating Axis Theorem (SAT) collision detection pipeline.

Three-phase collision engine:
1. Broadphase: Sort-and-Sweep AABB
2. Narrowphase: Float SAT with 1e-8 tolerance
3. Escalation: Certified interval arithmetic (50, 100, 200, 500 DPS) via mpmath.iv or Decimal
"""

from __future__ import annotations

import contextlib
import math
from decimal import Decimal, getcontext
from typing import List, Tuple, Sequence, Optional, Any

from src.geometry.corners import (
    Square,
    compute_corners_interval,
    aabb_overlap,
    HAS_MPMATH,
)

if HAS_MPMATH:
    import mpmath
    IVal = Any
else:
    mpmath = None
    IVal = Any

_ESCALATION_SEQUENCE = [50, 100, 200, 500]


@contextlib.contextmanager
def working_dps(dps: int):
    """Synchronize both mpmath.iv and mpmath.mp to dps digits (if mpmath available)."""
    if not HAS_MPMATH:
        yield
        return
    old_iv_dps = mpmath.iv.dps
    old_mp_dps = mpmath.mp.dps
    try:
        mpmath.iv.dps = dps
        mpmath.mp.dps = dps
        yield
    finally:
        mpmath.iv.dps = old_iv_dps
        mpmath.mp.dps = old_mp_dps


def sort_and_sweep(squares: List[Square], margin: float = 0.0) -> List[Tuple[int, int]]:
    """Sort squares along x-axis and return candidate overlapping pairs."""
    n = len(squares)
    if n <= 1:
        return []

    # Prepare sorted AABB records: (original_index, aabb)
    aabbs = [(i, sq.aabb()) for i, sq in enumerate(squares)]
    aabbs.sort(key=lambda item: item[1][0])

    pairs: List[Tuple[int, int]] = []
    for i_idx in range(n):
        i, aabb_i = aabbs[i_idx]
        for j_idx in range(i_idx + 1, n):
            j, aabb_j = aabbs[j_idx]
            # Spatial pruning along x-axis: if min_x of j > max_x of i + margin, break early
            if aabb_j[0] > aabb_i[2] + margin:
                break
            if not (
                aabb_i[2] + margin < aabb_j[0]
                or aabb_j[2] + margin < aabb_i[0]
                or aabb_i[3] + margin < aabb_j[1]
                or aabb_j[3] + margin < aabb_i[1]
            ):
                pairs.append((min(i, j), max(i, j)))

    # Deduplicate and sort pairs
    return sorted(list(set(pairs)))


def get_axes_float(angle1: float, angle2: float) -> List[Tuple[float, float]]:
    """Return the 4 candidate normal unit vectors from the two square rotations."""
    c1, s1 = math.cos(angle1), math.sin(angle1)
    c2, s2 = math.cos(angle2), math.sin(angle2)
    return [(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]


def project_corners_float(corners: List[Tuple[float, float]], ax: float, ay: float) -> Tuple[float, float]:
    """Project 4 corners onto axis (ax, ay) and return (min_proj, max_proj)."""
    projs = [ax * cx + ay * cy for cx, cy in corners]
    return (min(projs), max(projs))


def sat_check_float(
    sq1: Square,
    sq2: Square,
    undecided_tol: float = 1e-8,
) -> Tuple[str, float]:
    """Separating Axis Theorem test on 4 candidate axes.
    
    Returns:
        (status, min_overlap)
        status in ('separated', 'overlapping', 'undecided')
    """
    c1 = sq1.corners_float()
    c2 = sq2.corners_float()
    axes = get_axes_float(sq1.angle, sq2.angle)

    min_overlap = float("inf")
    for ax, ay in axes:
        min1, max1 = project_corners_float(c1, ax, ay)
        min2, max2 = project_corners_float(c2, ax, ay)
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < -undecided_tol:
            return "separated", overlap
        if overlap < min_overlap:
            min_overlap = overlap

    if min_overlap > undecided_tol:
        return "overlapping", min_overlap
    return "undecided", min_overlap


def iv_min(values: Sequence[Any]) -> Any:
    """Interval extension of min taking endpoint-wise bounds in mpmath.iv."""
    return mpmath.iv.mpf([min(v.a for v in values), min(v.b for v in values)])


def iv_max(values: Sequence[Any]) -> Any:
    """Interval extension of max taking endpoint-wise bounds in mpmath.iv."""
    return mpmath.iv.mpf([max(v.a for v in values), max(v.b for v in values)])


def iv_le(a: Any, b: Any) -> bool | None:
    """Certified less-or-equal: True if a <= b, False if a > b, None if indeterminate."""
    try:
        return a <= b
    except ValueError:
        return None


def sat_check_interval(
    sq1_corners: List[Tuple[Any, Any]],
    sq2_corners: List[Tuple[Any, Any]],
    angle1_iv: Any,
    angle2_iv: Any,
    safety_buffer: float = 0.0,
) -> bool | None:
    """Certified interval SAT on 4 normal axes using mpmath.iv.
    
    Returns:
        True: certified separated or non-penetrating contact (overlap <= 0 on at least one axis)
        False: certified overlapping (overlap > 0 on all axes)
        None: indeterminate comparison (needs higher DPS)
    """
    if not HAS_MPMATH:
        raise RuntimeError("mpmath is required for interval arithmetic SAT check")

    c1_iv, s1_iv = mpmath.iv.cos(angle1_iv), mpmath.iv.sin(angle1_iv)
    c2_iv, s2_iv = mpmath.iv.cos(angle2_iv), mpmath.iv.sin(angle2_iv)
    axes = [(c1_iv, s1_iv), (-s1_iv, c1_iv), (c2_iv, s2_iv), (-s2_iv, c2_iv)]

    has_undecided = False
    for ax, ay in axes:
        projs1 = [ax * cx + ay * cy for cx, cy in sq1_corners]
        projs2 = [ax * cx + ay * cy for cx, cy in sq2_corners]

        min1, max1 = iv_min(projs1), iv_max(projs1)
        min2, max2 = iv_min(projs2), iv_max(projs2)

        overlap = iv_min([max1, max2]) - iv_max([min1, min2])
        target_iv = (
            mpmath.iv.mpf(f"-{safety_buffer}") + mpmath.iv.mpf("1e-14")
            if safety_buffer > 0.0
            else mpmath.iv.mpf("0")
        )
        le_result = iv_le(overlap, target_iv)
        if le_result is True:
            return True
        if le_result is None:
            has_undecided = True

    if has_undecided:
        return None
    return False


def sat_check_decimal(
    sq1: Square,
    sq2: Square,
    dps: int = 50,
    safety_buffer: float = 0.0,
) -> bool:
    """High-precision arbitrary-precision check using Decimal for borderline contacts.
    
    Returns True if separated or non-penetrating (overlap <= 0 on at least 1 axis).
    Returns False if overlapping (> 0 on all 4 axes).
    """
    getcontext().prec = dps
    cx1 = Decimal(sq1.x_str if sq1.x_str else f"{sq1.x:.17g}")
    cy1 = Decimal(sq1.y_str if sq1.y_str else f"{sq1.y:.17g}")
    cx2 = Decimal(sq2.x_str if sq2.x_str else f"{sq2.x:.17g}")
    cy2 = Decimal(sq2.y_str if sq2.y_str else f"{sq2.y:.17g}")

    cos1 = Decimal(str(math.cos(sq1.angle)))
    sin1 = Decimal(str(math.sin(sq1.angle)))
    cos2 = Decimal(str(math.cos(sq2.angle)))
    sin2 = Decimal(str(math.sin(sq2.angle)))
    half = Decimal("0.5")

    dx1_1, dy1_1 = half * (cos1 - sin1), half * (sin1 + cos1)
    dx1_2, dy1_2 = -half * (cos1 + sin1), half * (-sin1 + cos1)
    c1 = [
        (cx1 + dx1_1, cy1 + dy1_1),
        (cx1 + dx1_2, cy1 + dy1_2),
        (cx1 - dx1_1, cy1 - dy1_1),
        (cx1 - dx1_2, cy1 - dy1_2),
    ]

    dx2_1, dy2_1 = half * (cos2 - sin2), half * (sin2 + cos2)
    dx2_2, dy2_2 = -half * (cos2 + sin2), half * (-sin2 + cos2)
    c2 = [
        (cx2 + dx2_1, cy2 + dy2_1),
        (cx2 + dx2_2, cy2 + dy2_2),
        (cx2 - dx2_1, cy2 - dy2_1),
        (cx2 - dx2_2, cy2 - dy2_2),
    ]

    target_ov = -Decimal(str(safety_buffer)) + Decimal("1e-14") if safety_buffer > 0.0 else Decimal("0")
    axes = [(cos1, sin1), (-sin1, cos1), (cos2, sin2), (-sin2, cos2)]
    for ax, ay in axes:
        p1 = [ax * x + ay * y for x, y in c1]
        p2 = [ax * x + ay * y for x, y in c2]
        overlap = min(max(p1), max(p2)) - max(min(p1), min(p2))
        if overlap <= target_ov:
            return True  # Certified separated / non-penetrating
    return False


def verify_puzzle_collision(
    squares: List[Square],
    safety_buffer: float = 0.0,
) -> Tuple[bool, str, List[Tuple[int, int]]]:
    """Run full broadphase + float SAT + interval arithmetic escalation on a list of squares.
    
    Returns:
        (is_feasible, reason, overlapping_pairs)
    """
    margin = max(1e-8, safety_buffer) if safety_buffer > 0.0 else 1e-8
    pairs = sort_and_sweep(squares, margin=margin)
    undecided_pairs: List[Tuple[int, int]] = []
    overlapping_pairs: List[Tuple[int, int]] = []

    tol = max(1e-8, safety_buffer)
    for i, j in pairs:
        status, min_ov = sat_check_float(squares[i], squares[j], undecided_tol=tol)
        if status == "overlapping":
            overlapping_pairs.append((i, j))
        elif status == "undecided":
            if safety_buffer > 0.0 and min_ov > -safety_buffer + 1e-14:
                # Violates required safety clearance buffer
                overlapping_pairs.append((i, j))
            else:
                undecided_pairs.append((i, j))
        elif status == "separated":
            if safety_buffer > 0.0 and min_ov > -safety_buffer + 1e-14:
                overlapping_pairs.append((i, j))

    if overlapping_pairs:
        return False, f"Squares {overlapping_pairs[0][0]} and {overlapping_pairs[0][1]} overlap (float check)", overlapping_pairs

    if undecided_pairs:
        if HAS_MPMATH:
            def check_certified(dps: int) -> bool | None:
                with working_dps(dps):
                    still_undecided = []
                    for i, j in undecided_pairs:
                        sq_i, sq_j = squares[i], squares[j]
                        cx_i = mpmath.iv.mpf(sq_i.x_str if sq_i.x_str else str(sq_i.x))
                        cy_i = mpmath.iv.mpf(sq_i.y_str if sq_i.y_str else str(sq_i.y))
                        ai = mpmath.iv.mpf(sq_i.angle_str if sq_i.angle_str else str(sq_i.angle))
                        cx_j = mpmath.iv.mpf(sq_j.x_str if sq_j.x_str else str(sq_j.x))
                        cy_j = mpmath.iv.mpf(sq_j.y_str if sq_j.y_str else str(sq_j.y))
                        aj = mpmath.iv.mpf(sq_j.angle_str if sq_j.angle_str else str(sq_j.angle))

                        corners_i = compute_corners_interval(cx_i, cy_i, ai)
                        corners_j = compute_corners_interval(cx_j, cy_j, aj)
                        res = sat_check_interval(corners_i, corners_j, ai, aj, safety_buffer=safety_buffer)
                        if res is False:
                            return False
                        if res is None:
                            still_undecided.append((i, j))
                    if still_undecided:
                        return None
                    return True

            certified = None
            for dps in _ESCALATION_SEQUENCE:
                certified = check_certified(dps)
                if certified is not None:
                    break

            if certified is False:
                return False, "Squares overlap (certified)", undecided_pairs
            if certified is None:
                return False, "Overlap undecidable at max precision", undecided_pairs
        else:
            # Decimal fallback across escalation sequence
            failed_pairs = []
            for i, j in undecided_pairs:
                is_sep = False
                for dps in _ESCALATION_SEQUENCE:
                    if sat_check_decimal(squares[i], squares[j], dps=dps, safety_buffer=safety_buffer):
                        is_sep = True
                        break
                if not is_sep:
                    failed_pairs.append((i, j))

            if failed_pairs:
                return False, f"Squares {failed_pairs[0][0]} and {failed_pairs[0][1]} overlap (certified decimal)", failed_pairs

    return True, "feasible", []
