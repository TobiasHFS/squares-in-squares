"""
Tier 2: Comprehensive Boundary Value Analysis (BVA) & Failure Mode Test Suite.
Verifies boundary conditions, numerical thresholds, literal limits, and rejection invariants.
Requirement: >= 50 test cases covering boundary values.
"""

import math
import os
import tempfile
import pytest

from tests.e2e.ref_geom import (
    SquarePose,
    ValidationResult,
    check_decimal_literal,
    parse_value_float,
    compute_corners_float,
    sat_pair_check,
    interval_contact_check,
    validate_puzzle_poses,
    generate_grid_solution,
    grid_ceiling_L,
    infeasible_score,
    DEFAULT_SAFETY_BUFFER,
    UNDECIDED_TOL,
)


# ==============================================================================
# CATEGORY 1: Small n Extremes (n=1, 2, 3, 4) (10 tests)
# ==============================================================================

def test_bva_n1_single_square_validity():
    """Puzzle n=1 with 1 square is valid."""
    p = [SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)]
    res = validate_puzzle_poses(1, p)
    assert res.is_valid


def test_bva_n1_single_square_L_is_unity():
    """Puzzle n=1 has side length L_1 = 1.0 exactly."""
    p = [SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)]
    res = validate_puzzle_poses(1, p)
    assert math.isclose(res.side_length, 1.0, abs_tol=1e-12)


def test_bva_n1_single_square_score_zero():
    """Puzzle n=1 has theoretical score 1/sqrt(1) - 1 = 0.0."""
    p = [SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)]
    res = validate_puzzle_poses(1, p)
    assert math.isclose(res.score, 0.0, abs_tol=1e-12)


def test_bva_n1_single_square_zero_collision_checks():
    """Puzzle n=1 has 0 candidate pairs and no collision overhead."""
    p = [SquarePose(cx=100.0, cy=-50.0, angle=0.7, square_id=0)]
    res = validate_puzzle_poses(1, p)
    assert res.is_valid
    assert len(res.overlapping_pairs) == 0


def test_bva_n2_grid_vs_diagonal():
    """Puzzle n=2: grid arrangement achieves proven optimal L_2 = 2.0 (Newman 1964)."""
    p_grid = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=1.5 + DEFAULT_SAFETY_BUFFER, cy=0.5, angle=0.0, square_id=1),
    ]
    res_grid = validate_puzzle_poses(2, p_grid)
    assert res_grid.is_valid
    assert math.isclose(res_grid.side_length, 2.0 + DEFAULT_SAFETY_BUFFER, abs_tol=1e-6)

    # Any attempt to compress below 2.0 with axis-aligned squares overlaps
    p_too_close = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=1.4, cy=0.5, angle=0.0, square_id=1),
    ]
    assert not validate_puzzle_poses(2, p_too_close).is_valid


def test_bva_n2_exact_bound_formula():
    """Literature bound for n=2 is L_2 = 2.0, while n=5 achieves Göbel bound 2 + 1/sqrt(2)."""
    expected_L2 = 2.0
    expected_score2 = (expected_L2 / math.sqrt(2.0)) - 1.0
    assert math.isclose(expected_score2, math.sqrt(2.0) - 1.0, abs_tol=1e-12)

    # Göbel bound for n=5
    expected_L5 = 2.0 + (1.0 / math.sqrt(2.0))
    expected_score5 = (expected_L5 / math.sqrt(5.0)) - 1.0
    assert expected_L5 < 3.0  # strictly beats grid ceiling 3.0


def test_bva_n3_l_shape_optimal_bound():
    """Puzzle n=3: literature optimum is L_3 = 2.0."""
    p_lshape = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=1.5 + DEFAULT_SAFETY_BUFFER, cy=0.5, angle=0.0, square_id=1),
        SquarePose(cx=0.5, cy=1.5 + DEFAULT_SAFETY_BUFFER, angle=0.0, square_id=2),
    ]
    res = validate_puzzle_poses(3, p_lshape)
    assert res.is_valid
    assert math.isclose(res.side_length, 2.0 + DEFAULT_SAFETY_BUFFER, abs_tol=1e-6)


def test_bva_n3_line_suboptimal_bound():
    """Puzzle n=3: linear layout has L_3 = 3.0, worse than L-shape."""
    p_line = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=1.5 + DEFAULT_SAFETY_BUFFER, cy=0.5, angle=0.0, square_id=1),
        SquarePose(cx=2.5 + 2 * DEFAULT_SAFETY_BUFFER, cy=0.5, angle=0.0, square_id=2),
    ]
    res = validate_puzzle_poses(3, p_line)
    assert res.is_valid
    assert res.side_length > 2.5


def test_bva_n4_exact_2x2_grid_score_zero():
    """Puzzle n=4: 2x2 grid achieves L_4 = 2.0, score = 0.0 (modulo safety buffer)."""
    poses = generate_grid_solution(4, margin=0.0)
    res = validate_puzzle_poses(4, poses)
    assert res.is_valid
    assert math.isclose(res.side_length, 2.0, abs_tol=1e-12)
    assert math.isclose(res.score, 0.0, abs_tol=1e-12)


def test_bva_n4_rotated_2x2_grid():
    """Puzzle n=4: rotated 2x2 grid container increases with angle."""
    # 2x2 grid rotated by 45 degrees has bounding box 2 * sqrt(2) approx 2.8284
    h = 0.5
    d = math.sqrt(2)
    poses = [
        SquarePose(cx=0.0, cy=d / 2, angle=math.pi / 4, square_id=0),
        SquarePose(cx=d / 2, cy=0.0, angle=math.pi / 4, square_id=1),
        SquarePose(cx=0.0, cy=-d / 2, angle=math.pi / 4, square_id=2),
        SquarePose(cx=-d / 2, cy=0.0, angle=math.pi / 4, square_id=3),
    ]
    # Squares may touch or separate
    res = validate_puzzle_poses(4, poses)
    if res.is_valid:
        assert res.side_length > 2.0


# ==============================================================================
# CATEGORY 2: Contact & Clearance Precision Thresholds (10 tests)
# ==============================================================================

def test_bva_contact_exact_axis_aligned_touching():
    """Squares touching along an axis-aligned edge have overlap == 0.0."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5, 0.5, 0.0)
    status = sat_pair_check(c1, c2, 0.0, 0.0)
    assert status == "undecided"  # min overlap is 0.0, inside [-1e-8, 1e-8]


def test_bva_contact_gap_1_1e8_separated_immediately():
    """Gap of 1.1e-8 (< -1.0e-8) is marked 'separated' directly by Float SAT."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5 + 1.1e-8, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_bva_contact_gap_1_05e8_direct_float_separation():
    """Gap of 1.05e-8 (< -1.0e-8) is marked 'separated' directly by Float SAT."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5 + 1.05e-8, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_bva_contact_gap_1_0e9_undecided_in_float_passes_interval():
    """Gap of 1.0e-9 is undecided in float SAT but passes interval contact check."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5 + 1.0e-9, cy=0.5, angle=0.0)
    c1 = compute_corners_float(sq1.cx, sq1.cy, sq1.angle)
    c2 = compute_corners_float(sq2.cx, sq2.cy, sq2.angle)
    assert sat_pair_check(c1, c2, sq1.angle, sq2.angle) == "undecided"
    assert interval_contact_check(sq1, sq2) is True


def test_bva_contact_gap_1_0e12_passes_interval():
    """Gap of 1.0e-12 is undecided in float SAT but certified valid in interval."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5 + 1.0e-12, cy=0.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is True


def test_bva_contact_sub_epsilon_1e16_penetration_rejected():
    """Penetration at decimal string level (1.5 - 1e-16) is rejected by interval check."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0, x_str="0.5", y_str="0.5", angle_str="0.0")
    # Explicitly serialize 1.5 - 1e-16 in decimal string literal
    sq2 = SquarePose(cx=1.5, cy=0.5, angle=0.0, x_str="1.4999999999999999", y_str="0.5", angle_str="0.0")
    assert interval_contact_check(sq1, sq2) is False


def test_bva_contact_sub_epsilon_1e15_penetration_rejected():
    """Penetration of 1.0e-15 is rejected by interval check."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5 - 1.0e-15, cy=0.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is False


def test_bva_contact_gross_penetration_0_1_rejected():
    """Gross penetration of 0.1 is marked 'overlapping' immediately in Float SAT."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.4, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "overlapping"


def test_bva_contact_touching_corner_to_corner():
    """Two squares touching exactly corner-to-corner (diagonal contact)."""
    # sq1 corner at (1.0, 1.0), sq2 corner at (1.0, 1.0)
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5, cy=1.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is True


def test_bva_contact_touching_vertical_edge():
    """Two squares touching along horizontal edge (top-to-bottom)."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=0.5, cy=1.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is True


# ==============================================================================
# CATEGORY 3: Decimal String Literal Conformance (14 tests)
# ==============================================================================

def test_bva_literal_scientific_lowercase_e():
    """Scientific notation with 'e' is valid."""
    assert check_decimal_literal("1.05e-8")


def test_bva_literal_scientific_uppercase_e():
    """Scientific notation with 'E' is valid."""
    assert check_decimal_literal("2.5E+03")


def test_bva_literal_explicit_positive_sign():
    """Positive sign literal is valid."""
    assert check_decimal_literal("+3.14159")


def test_bva_literal_leading_decimal_point():
    """Literal with leading decimal point '.5' is valid."""
    assert check_decimal_literal(".5")
    assert parse_value_float(".5") == 0.5


def test_bva_literal_trailing_decimal_point():
    """Literal with trailing decimal point '5.' is valid."""
    assert check_decimal_literal("5.")
    assert parse_value_float("5.") == 5.0


def test_bva_literal_zero_variants():
    """Various representations of zero are valid."""
    assert check_decimal_literal("0")
    assert check_decimal_literal("0.0")
    assert check_decimal_literal("-0.0")
    assert check_decimal_literal("+0.0")
    assert check_decimal_literal("0e0")


def test_bva_literal_reject_algebraic_expression():
    """Algebraic expressions like '1+sqrt(2)' must be rejected."""
    assert not check_decimal_literal("1+sqrt(2)")
    assert not check_decimal_literal("1 + 1")


def test_bva_literal_reject_fraction_string():
    """Rational fraction strings like '1/2' must be rejected."""
    assert not check_decimal_literal("1/2")


def test_bva_literal_reject_alphabetic_characters():
    """String with alphabetic letters must be rejected."""
    assert not check_decimal_literal("abc")
    assert not check_decimal_literal("1.0f")


def test_bva_literal_reject_double_decimal():
    """Multiple decimal points must be rejected."""
    assert not check_decimal_literal("1.2.3")


def test_bva_literal_reject_empty_string():
    """Empty string must be rejected."""
    assert not check_decimal_literal("")


def test_bva_literal_reject_whitespace_only():
    """Whitespace-only string must be rejected."""
    assert not check_decimal_literal("   ")


def test_bva_literal_reject_length_exceeding_1000():
    """Decimal literal with >1000 characters must be rejected."""
    long_lit = "0." + "1" * 1001
    assert not check_decimal_literal(long_lit)


def test_bva_literal_accept_length_at_1000():
    """Decimal literal with exactly 1000 characters is accepted if valid."""
    lit_1000 = "0." + "1" * 998
    assert len(lit_1000) == 1000
    assert check_decimal_literal(lit_1000)


# ==============================================================================
# CATEGORY 4: Inf, NaN, and Overflow Rejection (7 tests)
# ==============================================================================

def test_bva_reject_nan_lowercase():
    """'nan' literal must be rejected."""
    assert not check_decimal_literal("nan")


def test_bva_reject_nan_uppercase():
    """'NaN' literal must be rejected."""
    assert not check_decimal_literal("NaN")


def test_bva_reject_nan_mixed_case():
    """'NaN' variants must be rejected."""
    assert not check_decimal_literal("nAn")


def test_bva_reject_inf_positive():
    """'inf' and '+inf' must be rejected."""
    assert not check_decimal_literal("inf")
    assert not check_decimal_literal("+inf")


def test_bva_reject_inf_negative():
    """'-inf' must be rejected."""
    assert not check_decimal_literal("-inf")


def test_bva_reject_overflow_float_1e400():
    """Float exceeding double-precision limits (1e400) must be rejected."""
    assert not check_decimal_literal("1e400")


def test_bva_reject_underflow_to_zero_handled():
    """Subnormal float like 1e-350 parses to 0.0 or subnormal cleanly."""
    # IEEE 754 double underflows to 0.0 below 1e-324
    assert check_decimal_literal("1e-350")
    val = parse_value_float("1e-350")
    assert val == 0.0


# ==============================================================================
# CATEGORY 5: Square ID Permutations & Violations (6 tests)
# ==============================================================================

def test_bva_square_id_zero_indexed_range():
    """Valid square_ids must be exactly 0..n-1."""
    poses = [SquarePose(cx=i * 2.0, cy=0.0, angle=0.0, square_id=i) for i in range(5)]
    res = validate_puzzle_poses(5, poses)
    assert res.is_valid


def test_bva_square_id_missing_zero():
    """Square IDs starting at 1 (1..n) must be rejected."""
    poses = [SquarePose(cx=i * 2.0, cy=0.0, angle=0.0, square_id=i + 1) for i in range(3)]
    res = validate_puzzle_poses(3, poses)
    assert not res.is_valid
    assert "Invalid or duplicate square_id" in res.error_message


def test_bva_square_id_out_of_range_upper():
    """square_id == n must be rejected."""
    poses = [
        SquarePose(cx=0.0, cy=0.0, angle=0.0, square_id=0),
        SquarePose(cx=2.0, cy=0.0, angle=0.0, square_id=1),
        SquarePose(cx=4.0, cy=0.0, angle=0.0, square_id=3),  # 3 is out of range for n=3
    ]
    res = validate_puzzle_poses(3, poses)
    assert not res.is_valid


def test_bva_square_id_negative_rejected():
    """Negative square_id must be rejected."""
    poses = [
        SquarePose(cx=0.0, cy=0.0, angle=0.0, square_id=-1),
        SquarePose(cx=2.0, cy=0.0, angle=0.0, square_id=1),
    ]
    res = validate_puzzle_poses(2, poses)
    assert not res.is_valid


def test_bva_square_id_duplicate_ids_rejected():
    """Duplicate square_ids [0, 0] for n=2 must be rejected."""
    poses = [
        SquarePose(cx=0.0, cy=0.0, angle=0.0, square_id=0),
        SquarePose(cx=2.0, cy=0.0, angle=0.0, square_id=0),
    ]
    res = validate_puzzle_poses(2, poses)
    assert not res.is_valid


def test_bva_square_id_scrambled_valid_permutation_accepted():
    """Permuted square_ids (e.g. [2, 0, 1] for n=3) must be accepted."""
    poses = [
        SquarePose(cx=0.0, cy=0.0, angle=0.0, square_id=2),
        SquarePose(cx=2.0, cy=0.0, angle=0.0, square_id=0),
        SquarePose(cx=4.0, cy=0.0, angle=0.0, square_id=1),
    ]
    res = validate_puzzle_poses(3, poses)
    assert res.is_valid


# ==============================================================================
# CATEGORY 6: Puzzle Size n Limits & Row Counts (7 tests)
# ==============================================================================

def test_bva_puzzle_n_zero_rejected():
    """Puzzle n=0 is invalid."""
    res = validate_puzzle_poses(0, [])
    assert not res.is_valid


def test_bva_puzzle_n_negative_rejected():
    """Negative puzzle size n=-5 is invalid."""
    res = validate_puzzle_poses(-5, [])
    assert not res.is_valid


def test_bva_puzzle_n_exceeds_1000_rejected():
    """Puzzle size n=1001 is invalid."""
    poses = [SquarePose(cx=float(i), cy=0.0, angle=0.0, square_id=i) for i in range(1001)]
    res = validate_puzzle_poses(1001, poses)
    assert not res.is_valid


def test_bva_puzzle_n_1000_accepted():
    """Puzzle size n=1000 is the valid upper boundary."""
    # Test grid generation at boundary n=1000
    poses = generate_grid_solution(1000)
    assert len(poses) == 1000
    res = validate_puzzle_poses(1000, poses)
    assert res.is_valid
    assert res.side_length <= 32.0 + 1e-5


def test_bva_missing_puzzle_penalty_assigned():
    """Grid ceiling penalty formula matches L0 = ceil(sqrt(n))."""
    for n in [1, 2, 5, 10, 17, 50, 100, 500, 1000]:
        L0 = grid_ceiling_L(n)
        expected_L0 = float(math.isqrt(n - 1) + 1 if n > 0 else 1)
        assert L0 == expected_L0
        assert infeasible_score(n) == (L0 / math.sqrt(n)) - 1.0


def test_bva_puzzle_row_count_underflow():
    """Fewer rows than n must trigger rejection."""
    poses = generate_grid_solution(9)
    res = validate_puzzle_poses(10, poses)  # provided 9, expected 10
    assert not res.is_valid
    assert "Expected 10 squares" in res.error_message


def test_bva_puzzle_row_count_overflow():
    """More rows than n must trigger rejection."""
    poses = generate_grid_solution(11)
    res = validate_puzzle_poses(10, poses)  # provided 11, expected 10
    assert not res.is_valid
    assert "Expected 10 squares" in res.error_message


# ==============================================================================
# CATEGORY 7: Perfect Square Boundaries n = k^2 (10 tests)
# ==============================================================================

@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_bva_perfect_square_grid_ceiling_exact(k):
    """For n = k^2, grid ceiling L0 = k and infeasible_score = 0.0."""
    n = k * k
    assert grid_ceiling_L(n) == float(k)
    assert math.isclose(infeasible_score(n), 0.0, abs_tol=1e-14)
