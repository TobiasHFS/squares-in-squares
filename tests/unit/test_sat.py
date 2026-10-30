"""Unit tests for src/geometry/sat.py."""

import math
import pytest
from src.geometry.corners import Square
from src.geometry.sat import (
    sort_and_sweep,
    get_axes_float,
    project_corners_float,
    sat_check_float,
    sat_check_decimal,
    verify_puzzle_collision,
)


def test_broadphase_sweep_empty_or_single():
    assert sort_and_sweep([]) == []
    assert sort_and_sweep([Square(x=0.0, y=0.0, angle=0.0)]) == []


def test_broadphase_sweep_prunes_distant_squares():
    sq1 = Square(x=0.0, y=0.0, angle=0.0)
    sq2 = Square(x=5.0, y=0.0, angle=0.0)
    assert sort_and_sweep([sq1, sq2]) == []


def test_broadphase_sweep_detects_adjacent_squares():
    sq1 = Square(x=0.0, y=0.0, angle=0.0)
    sq2 = Square(x=0.9, y=0.0, angle=0.0)
    pairs = sort_and_sweep([sq1, sq2])
    assert pairs == [(0, 1)]


def test_sat_separated_clearly():
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=2.5, y=0.5, angle=0.0)
    status, ov = sat_check_float(sq1, sq2, undecided_tol=1e-8)
    assert status == "separated"
    assert ov < -0.9


def test_sat_overlapping_clearly():
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=0.8, y=0.5, angle=0.0)
    status, ov = sat_check_float(sq1, sq2, undecided_tol=1e-8)
    assert status == "overlapping"
    assert ov > 0.6


def test_sat_borderline_touching():
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=1.5, y=0.5, angle=0.0)  # theoretical separation 0.0
    status, ov = sat_check_float(sq1, sq2, undecided_tol=1e-8)
    assert status == "undecided"
    assert math.isclose(ov, 0.0, abs_tol=1e-12)


def test_sat_safety_buffer_clears_undecided():
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=1.5 + 1.1e-8, y=0.5, angle=0.0)
    status, ov = sat_check_float(sq1, sq2, undecided_tol=1e-8)
    assert status == "separated"
    assert ov < -1e-8


def test_sat_check_decimal_boundary_contact():
    """Exact edge-to-edge contact is certified non-penetrating in decimal escalation."""
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=1.5, y=0.5, angle=0.0)
    assert sat_check_decimal(sq1, sq2, dps=50) is True


def test_sat_check_decimal_penetration():
    """True overlap of 1e-6 fails certified decimal check."""
    sq1 = Square(x=0.5, y=0.5, angle=0.0)
    sq2 = Square(x=1.5 - 1e-6, y=0.5, angle=0.0)
    assert sat_check_decimal(sq1, sq2, dps=50) is False


def test_verify_puzzle_collision_feasible():
    squares = [
        Square(x=0.5, y=0.5, angle=0.0),
        Square(x=1.6, y=0.5, angle=0.0),
        Square(x=0.5, y=1.6, angle=0.0),
        Square(x=1.6, y=1.6, angle=0.0),
    ]
    is_feas, reason, ov_pairs = verify_puzzle_collision(squares)
    assert is_feas is True
    assert ov_pairs == []


def test_verify_puzzle_collision_infeasible():
    squares = [
        Square(x=0.5, y=0.5, angle=0.0),
        Square(x=0.8, y=0.5, angle=0.0),  # overlaps with square 0
    ]
    is_feas, reason, ov_pairs = verify_puzzle_collision(squares)
    assert is_feas is False
    assert (0, 1) in ov_pairs
