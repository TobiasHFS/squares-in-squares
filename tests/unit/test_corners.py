"""Unit tests for src/geometry/corners.py."""

import math
import pytest
from src.geometry.corners import (
    SquarePose,
    Square,
    compute_corners_float,
    compute_aabb,
    aabb_overlap,
    compute_bounding_container_float,
    measure_side_mp,
)


def test_square_corners_unrotated():
    """Unit square at (0.5, 0.5) with angle 0 has corners (1,1), (0,1), (0,0), (1,0)."""
    corners = compute_corners_float(0.5, 0.5, 0.0)
    assert len(corners) == 4
    # Corners: v0=(1,1), v1=(0,1), v2=(0,0), v3=(1,0)
    expected = [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]
    for (cx, cy), (ex, ey) in zip(corners, expected):
        assert math.isclose(cx, ex, abs_tol=1e-12)
        assert math.isclose(cy, ey, abs_tol=1e-12)


def test_square_corners_45_degree():
    """Square rotated by 45 degrees: distance from center to all 4 corners is 1/sqrt(2)."""
    angle = math.pi / 4
    cx, cy = 2.0, 3.0
    corners = compute_corners_float(cx, cy, angle)
    expected_dist = math.sqrt(0.5)
    for px, py in corners:
        dist = math.hypot(px - cx, py - cy)
        assert math.isclose(dist, expected_dist, abs_tol=1e-12)


def test_square_corners_periodicity():
    """Angle theta and theta + 2*pi produce identical corners."""
    c1 = compute_corners_float(1.5, -2.5, 0.73)
    c2 = compute_corners_float(1.5, -2.5, 0.73 + 2 * math.pi)
    for (x1, y1), (x2, y2) in zip(c1, c2):
        assert math.isclose(x1, x2, abs_tol=1e-12)
        assert math.isclose(y1, y2, abs_tol=1e-12)


def test_compute_aabb():
    """AABB of rotated square correctly encloses all 4 corners."""
    corners = compute_corners_float(1.0, 1.0, math.pi / 4)
    min_x, min_y, max_x, max_y = compute_aabb(corners)
    for x, y in corners:
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
    # For angle 45 deg, radius is sqrt(2)/2
    r = math.sqrt(0.5)
    assert math.isclose(min_x, 1.0 - r, abs_tol=1e-12)
    assert math.isclose(max_x, 1.0 + r, abs_tol=1e-12)


def test_aabb_overlap():
    """aabb_overlap returns True for overlapping and False for separated boxes."""
    b1 = (0.0, 0.0, 2.0, 2.0)
    b2 = (1.5, 1.5, 3.0, 3.0)
    b3 = (2.1, 0.0, 4.0, 2.0)
    assert aabb_overlap(b1, b2) is True
    assert aabb_overlap(b1, b3) is False


def test_container_side_length_translation_invariance():
    """Container side length Ln is invariant under translation."""
    poses = [
        Square(x=0.5, y=0.5, angle=0.0),
        Square(x=1.6, y=0.5, angle=0.2),
        Square(x=0.5, y=1.7, angle=0.0),
    ]
    c_orig = [sq.corners_float() for sq in poses]
    L_orig = compute_bounding_container_float(c_orig)

    # Translate by (+50.0, -100.0)
    poses_shifted = [
        Square(x=sq.x + 50.0, y=sq.y - 100.0, angle=sq.angle)
        for sq in poses
    ]
    c_shifted = [sq.corners_float() for sq in poses_shifted]
    L_shifted = compute_bounding_container_float(c_shifted)

    assert math.isclose(L_orig, L_shifted, abs_tol=1e-12)


def test_measure_side_mp():
    """measure_side_mp matches compute_bounding_container_float for float precision."""
    squares = [
        Square(x=0.5, y=0.5, angle=0.0),
        Square(x=1.5, y=0.5, angle=0.0),
        Square(x=0.5, y=1.5, angle=0.0),
        Square(x=1.5, y=1.5, angle=0.0),
    ]
    side = measure_side_mp(squares)
    assert math.isclose(side, 2.0, abs_tol=1e-9)
