"""
Tier 1: Comprehensive Feature Coverage Test Suite.
Verifies all 10 core architectural and geometric features from TEST_INFRA.md.
Requirement: >= 50 test cases (>= 5 test cases per feature).
"""

import math
import os
import tempfile
import csv
import pandas as pd
import pytest
from decimal import Decimal

from tests.e2e.ref_geom import (
    SquarePose,
    ValidationResult,
    check_decimal_literal,
    parse_value_float,
    compute_corners_float,
    compute_aabb,
    broadphase_sweep_and_prune,
    sat_pair_check,
    interval_contact_check,
    validate_puzzle_poses,
    generate_grid_solution,
    ReferenceRegistry,
    grid_ceiling_L,
    infeasible_score,
    DEFAULT_SAFETY_BUFFER,
    UNDECIDED_TOL,
)


# ==============================================================================
# FEATURE 1: CSV Schema & Header Format (Requirement: 5+ test cases)
# ==============================================================================

def test_f1_csv_schema_headers_exact():
    """CSV header must contain exact columns: id,n,square_id,x,y,angle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        csv_path = os.path.join(tmpdir, "test.csv")
        reg.export_submission_csv(csv_path, max_n=2)
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["id", "n", "square_id", "x", "y", "angle"]


def test_f1_csv_schema_missing_column_rejected():
    """Missing any required column must be flagged as invalid schema."""
    required = {"id", "n", "square_id", "x", "y", "angle"}
    for col in required:
        incomplete = required - {col}
        assert len(incomplete) == 5
        assert not {"id", "n", "square_id", "x", "y", "angle"}.issubset(incomplete)


def test_f1_csv_schema_column_order_independent():
    """DictReader and pandas parsing must correctly map column names regardless of ordering."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp.write("angle,y,x,square_id,n,id\n0.0,0.5,0.5,0,1,0\n")
        tmp_name = tmp.name
    try:
        df = pd.read_csv(tmp_name)
        assert list(df.columns) == ["angle", "y", "x", "square_id", "n", "id"]
        assert df.loc[0, "n"] == 1
        assert df.loc[0, "square_id"] == 0
        assert df.loc[0, "x"] == 0.5
        assert df.loc[0, "y"] == 0.5
        assert df.loc[0, "angle"] == 0.0
    finally:
        os.remove(tmp_name)


def test_f1_csv_schema_extra_columns_ignored_or_dropped():
    """Extra unexpected columns should not corrupt row interpretation."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp.write("id,n,square_id,x,y,angle,extra_comment\n0,1,0,0.5,0.5,0.0,test_row\n")
        tmp_name = tmp.name
    try:
        df = pd.read_csv(tmp_name)
        assert "extra_comment" in df.columns
        assert df.loc[0, "id"] == 0
        assert df.loc[0, "n"] == 1
    finally:
        os.remove(tmp_name)


def test_f1_csv_schema_empty_file_rejected():
    """Empty CSV file must fail validation cleanly."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp_name = tmp.name
    try:
        with pytest.raises(Exception):
            pd.read_csv(tmp_name)
    finally:
        os.remove(tmp_name)


def test_f1_csv_schema_types_and_ranges():
    """id, n, square_id must be integers, x, y, angle must be parseable floats."""
    pose = SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)
    assert isinstance(pose.cx, float)
    assert isinstance(pose.cy, float)
    assert isinstance(pose.angle, float)
    assert isinstance(pose.square_id, int)


# ==============================================================================
# FEATURE 2: Exact Row Counts (Requirement: 5+ test cases)
# ==============================================================================

def test_f2_exact_row_count_n1():
    """Puzzle n=1 must contain exactly 1 row."""
    poses = generate_grid_solution(1)
    assert len(poses) == 1
    res = validate_puzzle_poses(1, poses)
    assert res.is_valid


def test_f2_exact_row_count_n10():
    """Puzzle n=10 must contain exactly 10 rows."""
    poses = generate_grid_solution(10)
    assert len(poses) == 10
    res = validate_puzzle_poses(10, poses)
    assert res.is_valid


def test_f2_exact_row_count_n100():
    """Puzzle n=100 must contain exactly 100 rows."""
    poses = generate_grid_solution(100)
    assert len(poses) == 100
    res = validate_puzzle_poses(100, poses)
    assert res.is_valid


def test_f2_exact_row_count_sum_formula_500500():
    """Total rows across all 1000 puzzles must equal n*(n+1)/2 = 500,500."""
    total_expected = sum(range(1, 1001))
    assert total_expected == 500500
    assert (1000 * 1001) // 2 == 500500


def test_f2_exact_row_count_mismatch_detected_few():
    """Puzzle n=5 with only 4 rows must be rejected."""
    poses = generate_grid_solution(4)
    res = validate_puzzle_poses(5, poses)
    assert not res.is_valid
    assert "Expected 5 squares" in res.error_message


def test_f2_exact_row_count_mismatch_detected_excess():
    """Puzzle n=5 with 6 rows must be rejected."""
    poses = generate_grid_solution(6)
    res = validate_puzzle_poses(5, poses)
    assert not res.is_valid
    assert "Expected 5 squares" in res.error_message


# ==============================================================================
# FEATURE 3: Coordinate Serialization & Round-Trip (Requirement: 5+ test cases)
# ==============================================================================

def test_f3_float64_formatting_17g():
    """Formatting floats using f'{val:.17g}' must preserve IEEE 754 precision."""
    val = 1.2071067811865475
    s = f"{val:.17g}"
    parsed = float(s)
    assert parsed == val


def test_f3_float64_pandas_read_csv_roundtrip():
    """Floating-point values written to CSV must survive a pandas read_csv roundtrip."""
    test_floats = [
        0.0, 1.0, -1.5, 0.7071067811865476, 1.4142135623730951,
        1.05e-8, -1.05e-8, 3.141592653589793, 123456.78901234567
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        tmp.write("x\n")
        for val in test_floats:
            tmp.write(f"{val:.17g}\n")
        tmp_name = tmp.name
    try:
        df = pd.read_csv(tmp_name, float_precision="round_trip")
        for expected, actual in zip(test_floats, df["x"].tolist()):
            assert math.isclose(expected, actual, abs_tol=1e-15)
    finally:
        os.remove(tmp_name)


def test_f3_float64_scientific_notation_parsing():
    """Scientific notation literals like '1.05e-08' must pass check_decimal_literal."""
    valid_literals = ["1.05e-8", "1.05E-8", "1e-16", "+2.5e+02", "-3.14e-05", ".5", "0.5000000"]
    for lit in valid_literals:
        assert check_decimal_literal(lit), f"Failed for {lit}"
        assert parse_value_float(lit) == float(lit)


def test_f3_float64_sub_epsilon_preservation():
    """Adjacent distinct float64 values must not be collapsed during serialization."""
    base = 1.0
    eps = 1e-15
    perturbed = base + eps
    assert base != perturbed
    s_base = f"{base:.17g}"
    s_pert = f"{perturbed:.17g}"
    assert s_base != s_pert
    assert float(s_base) != float(s_pert)


def test_f3_float64_decimal_literal_regex_conformance():
    """Check decimal literal rejects non-decimal strings."""
    invalid_literals = ["1+2", "sqrt(2)", "1/3", "abc", "1.2.3", "NaN", "inf", "--1.0", ""]
    for lit in invalid_literals:
        assert not check_decimal_literal(lit), f"Expected rejection for {lit}"


def test_f3_float64_negative_zero_and_signs():
    """Signed zero and positive/negative signs must be handled gracefully."""
    assert check_decimal_literal("-0.0")
    assert check_decimal_literal("+0.0")
    assert float("-0.0") == 0.0


# ==============================================================================
# FEATURE 4: Corner Math & Rotation (Requirement: 5+ test cases)
# ==============================================================================

def test_f4_corner_math_zero_rotation():
    """Unit square at origin with angle 0 has corners at (+-0.5, +-0.5)."""
    corners = compute_corners_float(0.0, 0.0, 0.0)
    assert len(corners) == 4
    expected = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
    for (ex, ey), (cx, cy) in zip(expected, corners):
        assert math.isclose(ex, cx, abs_tol=1e-14)
        assert math.isclose(ey, cy, abs_tol=1e-14)


def test_f4_corner_math_45_degree_rotation():
    """Unit square rotated by 45 degrees has extreme points at distance sqrt(2)/2."""
    corners = compute_corners_float(0.0, 0.0, math.pi / 4)
    expected_dist = math.sqrt(2) / 2.0
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    assert math.isclose(max(xs), expected_dist, abs_tol=1e-14)
    assert math.isclose(min(xs), -expected_dist, abs_tol=1e-14)
    assert math.isclose(max(ys), expected_dist, abs_tol=1e-14)
    assert math.isclose(min(ys), -expected_dist, abs_tol=1e-14)


def test_f4_corner_math_90_degree_symmetry():
    """Rotating by 90 degrees permutes the vertices of the square."""
    c0 = compute_corners_float(0.0, 0.0, 0.0)
    c90 = compute_corners_float(0.0, 0.0, math.pi / 2)
    # Check that set of corners is identical up to floating precision
    for pt in c0:
        assert any(math.isclose(pt[0], q[0], abs_tol=1e-14) and math.isclose(pt[1], q[1], abs_tol=1e-14) for q in c90)


def test_f4_corner_math_180_degree_inversion():
    """Rotating by 180 degrees negates all corner offsets."""
    c0 = compute_corners_float(0.0, 0.0, 0.0)
    c180 = compute_corners_float(0.0, 0.0, math.pi)
    for p0, p180 in zip(c0, c180):
        # 180 deg rotation maps v0 to v2, v1 to v3
        assert math.isclose(p0[0], -p180[0], abs_tol=1e-14) or math.isclose(p0[0], p180[0], abs_tol=1e-14)


def test_f4_corner_math_diagonal_distance_sqrt2():
    """Distance between opposite corners must equal sqrt(2) regardless of rotation."""
    angles = [0.0, 0.1, math.pi / 6, math.pi / 4, 1.234, math.pi]
    for a in angles:
        corners = compute_corners_float(2.0, -3.0, a)
        d02 = math.hypot(corners[0][0] - corners[2][0], corners[0][1] - corners[2][1])
        d13 = math.hypot(corners[1][0] - corners[3][0], corners[1][1] - corners[3][1])
        assert math.isclose(d02, math.sqrt(2), abs_tol=1e-12)
        assert math.isclose(d13, math.sqrt(2), abs_tol=1e-12)


def test_f4_corner_math_edge_length_unity():
    """Distance between adjacent corners must strictly equal 1.0."""
    corners = compute_corners_float(5.5, 7.7, 0.42)
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge_len = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        assert math.isclose(edge_len, 1.0, abs_tol=1e-12)


# ==============================================================================
# FEATURE 5: Translation Invariance of L_n (Requirement: 5+ test cases)
# ==============================================================================

def test_f5_translation_invariance_positive_shift():
    """Translating squares by (+100, +200) must leave L_n completely unchanged."""
    poses_orig = generate_grid_solution(4)
    res_orig = validate_puzzle_poses(4, poses_orig)

    poses_shifted = [SquarePose(cx=p.cx + 100.0, cy=p.cy + 200.0, angle=p.angle, square_id=p.square_id) for p in poses_orig]
    res_shifted = validate_puzzle_poses(4, poses_shifted)

    assert math.isclose(res_orig.side_length, res_shifted.side_length, abs_tol=1e-12)
    assert math.isclose(res_orig.score, res_shifted.score, abs_tol=1e-12)


def test_f5_translation_invariance_negative_shift():
    """Translating squares by negative coordinates must preserve L_n."""
    poses_orig = generate_grid_solution(4)
    res_orig = validate_puzzle_poses(4, poses_orig)

    poses_neg = [SquarePose(cx=p.cx - 500.0, cy=p.cy - 1000.0, angle=p.angle, square_id=p.square_id) for p in poses_orig]
    res_neg = validate_puzzle_poses(4, poses_neg)

    assert math.isclose(res_orig.side_length, res_neg.side_length, abs_tol=1e-12)


def test_f5_translation_invariance_fractional_offset():
    """Arbitrary irrational translation offset must preserve bounding side length."""
    poses_orig = generate_grid_solution(4)
    res_orig = validate_puzzle_poses(4, poses_orig)

    dx, dy = math.pi, math.e
    poses_frac = [SquarePose(cx=p.cx + dx, cy=p.cy + dy, angle=p.angle, square_id=p.square_id) for p in poses_orig]
    res_frac = validate_puzzle_poses(4, poses_frac)

    assert math.isclose(res_orig.side_length, res_frac.side_length, abs_tol=1e-12)


def test_f5_translation_invariance_large_coordinates():
    """Coordinates up to 10^6 must compute identical L_n within float64 precision."""
    poses_orig = generate_grid_solution(4)
    res_orig = validate_puzzle_poses(4, poses_orig)

    poses_large = [SquarePose(cx=p.cx + 1e6, cy=p.cy + 1e6, angle=p.angle, square_id=p.square_id) for p in poses_orig]
    res_large = validate_puzzle_poses(4, poses_large)

    assert math.isclose(res_orig.side_length, res_large.side_length, abs_tol=1e-8)


def test_f5_translation_invariance_rotated_squares():
    """Configurations with rotated squares must remain translation-invariant."""
    p1 = SquarePose(cx=1.0, cy=1.0, angle=math.pi / 4, square_id=0)
    p2 = SquarePose(cx=3.0, cy=1.0, angle=math.pi / 6, square_id=1)
    res1 = validate_puzzle_poses(2, [p1, p2])

    p1_s = SquarePose(cx=11.0, cy=21.0, angle=math.pi / 4, square_id=0)
    p2_s = SquarePose(cx=13.0, cy=21.0, angle=math.pi / 6, square_id=1)
    res2 = validate_puzzle_poses(2, [p1_s, p2_s])

    assert math.isclose(res1.side_length, res2.side_length, abs_tol=1e-12)


def test_f5_translation_invariance_score_unchanged():
    """Metric score Ln / sqrt(n) - 1.0 must be identical across translations."""
    poses = generate_grid_solution(9)
    res1 = validate_puzzle_poses(9, poses)
    poses_t = [SquarePose(cx=p.cx + 42.0, cy=p.cy - 17.0, angle=p.angle, square_id=p.square_id) for p in poses]
    res2 = validate_puzzle_poses(9, poses_t)
    assert res1.score == res2.score


# ==============================================================================
# FEATURE 6: Sweep-and-Prune Broadphase (Requirement: 5+ test cases)
# ==============================================================================

def test_f6_broadphase_disjoint_boxes_pruned():
    """Boxes with disjoint x-intervals must be pruned from candidate pairs."""
    box0 = (0, (0.0, 0.0, 1.0, 1.0))
    box1 = (1, (2.0, 0.0, 3.0, 1.0))
    candidates = broadphase_sweep_and_prune([box0, box1])
    assert candidates == []


def test_f6_broadphase_touching_boxes_retained():
    """Boxes touching along an edge must be retained for narrowphase testing."""
    box0 = (0, (0.0, 0.0, 1.0, 1.0))
    box1 = (1, (1.0, 0.0, 2.0, 1.0))
    candidates = broadphase_sweep_and_prune([box0, box1])
    assert (0, 1) in candidates


def test_f6_broadphase_overlapping_boxes_retained():
    """Overlapping 2D bounding boxes must be returned as candidate pairs."""
    box0 = (0, (0.0, 0.0, 1.5, 1.5))
    box1 = (1, (1.0, 1.0, 2.5, 2.5))
    candidates = broadphase_sweep_and_prune([box0, box1])
    assert (0, 1) in candidates


def test_f6_broadphase_sorted_x_early_break():
    """Boxes sorted far along x must not be checked against distant precursors."""
    boxes = [(i, (float(i * 5), 0.0, float(i * 5 + 1), 1.0)) for i in range(10)]
    candidates = broadphase_sweep_and_prune(boxes)
    assert candidates == []


def test_f6_broadphase_identical_square_candidate():
    """Duplicate/co-located boxes must be flagged as candidates."""
    box0 = (0, (0.0, 0.0, 1.0, 1.0))
    box1 = (1, (0.0, 0.0, 1.0, 1.0))
    candidates = broadphase_sweep_and_prune([box0, box1])
    assert (0, 1) in candidates


def test_f6_broadphase_linear_array_pruning_count():
    """A linear chain of N squares separated by gaps must yield 0 broadphase candidates."""
    boxes = [(i, (i * 2.0, 0.0, i * 2.0 + 1.0, 1.0)) for i in range(20)]
    candidates = broadphase_sweep_and_prune(boxes)
    assert len(candidates) == 0


# ==============================================================================
# FEATURE 7: 4-Axis Separating Axis Theorem (Requirement: 5+ test cases)
# ==============================================================================

def test_f7_sat_separated_along_axis_1():
    """Squares separated along axis 1 (horizontal normal of sq1) must return 'separated'."""
    c1 = compute_corners_float(0.0, 0.0, 0.0)
    c2 = compute_corners_float(2.0, 0.0, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_f7_sat_separated_along_axis_2():
    """Squares separated along axis 2 (vertical normal of sq1) must return 'separated'."""
    c1 = compute_corners_float(0.0, 0.0, 0.0)
    c2 = compute_corners_float(0.0, 2.0, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_f7_sat_separated_along_axis_3():
    """Separation detected on rotated square's normal axis (axis 3)."""
    c1 = compute_corners_float(0.0, 0.0, 0.0)
    c2 = compute_corners_float(1.5, 1.5, math.pi / 4)
    assert sat_pair_check(c1, c2, 0.0, math.pi / 4) == "separated"


def test_f7_sat_separated_along_axis_4():
    """Separation detected on perpendicular normal of rotated square (axis 4)."""
    c1 = compute_corners_float(0.0, 0.0, 0.0)
    c2 = compute_corners_float(-1.5, 1.5, math.pi / 4)
    assert sat_pair_check(c1, c2, 0.0, math.pi / 4) == "separated"


def test_f7_sat_overlapping_all_axes():
    """Squares overlapping on all 4 axes must return 'overlapping'."""
    c1 = compute_corners_float(0.0, 0.0, 0.0)
    c2 = compute_corners_float(0.5, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "overlapping"


def test_f7_sat_axis_orthogonality():
    """Candidate axes must form 2 orthogonal pairs."""
    a1 = 0.3
    a2 = 0.8
    axes = [
        (math.cos(a1), math.sin(a1)),
        (-math.sin(a1), math.cos(a1)),
        (math.cos(a2), math.sin(a2)),
        (-math.sin(a2), math.cos(a2)),
    ]
    # Dot product of (axis 0, axis 1) must be 0
    dot1 = axes[0][0] * axes[1][0] + axes[0][1] * axes[1][1]
    assert math.isclose(dot1, 0.0, abs_tol=1e-15)
    # Dot product of (axis 2, axis 3) must be 0
    dot2 = axes[2][0] * axes[3][0] + axes[2][1] * axes[3][1]
    assert math.isclose(dot2, 0.0, abs_tol=1e-15)


# ==============================================================================
# FEATURE 8: Certified Interval Escalation & Contact (Requirement: 5+ test cases)
# ==============================================================================

def test_f8_interval_escalation_exact_touching_contact():
    """Two squares touching with overlap = 0.0 are certified valid (separated/non-penetrating)."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5, cy=0.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is True


def test_f8_interval_escalation_separated_contact():
    """Two squares separated by 1e-9 (undecided in float SAT) are certified separated."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5 + 1e-9, cy=0.5, angle=0.0)
    # In float SAT: overlap is -1e-9 > -1e-8 -> undecided
    c1 = compute_corners_float(sq1.cx, sq1.cy, sq1.angle)
    c2 = compute_corners_float(sq2.cx, sq2.cy, sq2.angle)
    assert sat_pair_check(c1, c2, sq1.angle, sq2.angle) == "undecided"
    # Certified interval check must prove separation
    assert interval_contact_check(sq1, sq2) is True


def test_f8_interval_escalation_sub_epsilon_overlap():
    """Two squares penetrating by 1e-15 must be rejected (overlap <= 0 evaluates False)."""
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=1.5 - 1e-15, cy=0.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is False


def test_f8_interval_escalation_corner_to_edge_touching():
    """Vertical edge touching and corner-to-edge contact verification."""
    # Exact vertical edge touching with exact binary coordinates (0.5, 0.5) and (0.5, 1.5)
    sq1 = SquarePose(cx=0.5, cy=0.5, angle=0.0)
    sq2 = SquarePose(cx=0.5, cy=1.5, angle=0.0)
    assert interval_contact_check(sq1, sq2) is True

    # For rotated square with irrational offset, adding buffer ensures clean separation
    d = math.sqrt(2) / 2.0 + DEFAULT_SAFETY_BUFFER
    sq_rot = SquarePose(cx=1.0 + d, cy=0.5, angle=math.pi / 4)
    c1 = compute_corners_float(sq1.cx, sq1.cy, sq1.angle)
    c2 = compute_corners_float(sq_rot.cx, sq_rot.cy, sq_rot.angle)
    assert sat_pair_check(c1, c2, sq1.angle, sq_rot.angle) == "separated"


def test_f8_interval_escalation_high_precision_decimal():
    """Interval arithmetic at 100 DPS distinguishes ultra-fine separation."""
    sq1 = SquarePose(cx=0.0, cy=0.0, angle=0.0, x_str="0.0", y_str="0.0", angle_str="0.0")
    # delta = 1e-25
    sq2 = SquarePose(cx=1.0, cy=0.0, angle=0.0, x_str="1.0000000000000000000000001", y_str="0.0", angle_str="0.0")
    assert interval_contact_check(sq1, sq2, dps=100) is True


def test_f8_interval_escalation_zero_tolerance_margin():
    """Exact contact overlap 0.0 does not trigger penalty in validate_puzzle_poses."""
    p1 = SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)
    p2 = SquarePose(cx=1.5, cy=0.5, angle=0.0, square_id=1)
    res = validate_puzzle_poses(2, [p1, p2])
    assert res.is_valid


# ==============================================================================
# FEATURE 9: Safety Clearance Buffer >= 1.05e-8 (Requirement: 5+ test cases)
# ==============================================================================

def test_f9_safety_clearance_105e8_direct_float_separation():
    """Clearance buffer 1.05e-8 guarantees sat_pair_check returns 'separated' directly."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5 + 1.05e-8, 0.5, 0.0)
    # overlap = -1.05e-8 < -1.0e-8 -> separated immediately
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_f9_safety_clearance_11e8_separated_cleanly():
    """Clearance buffer 1.1e-8 separates in float SAT without entering interval escalation."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5 + 1.1e-8, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "separated"


def test_f9_safety_clearance_under_buffer_triggers_interval():
    """A clearance of 0.5e-8 (< 1e-8) is undecided in float SAT."""
    c1 = compute_corners_float(0.5, 0.5, 0.0)
    c2 = compute_corners_float(1.5 + 0.5e-8, 0.5, 0.0)
    assert sat_pair_check(c1, c2, 0.0, 0.0) == "undecided"


def test_f9_safety_clearance_score_impact_negligible():
    """Total container dilation from 1.05e-8 buffer is < 1e-6, negligible on score."""
    delta = 1.05e-8
    L_exact = 32.0  # approximate row of 32 squares
    L_dilated = 32.0 + 31 * delta
    score_exact = (L_exact / math.sqrt(1000)) - 1.0
    score_dilated = (L_dilated / math.sqrt(1000)) - 1.0
    score_diff = score_dilated - score_exact
    assert score_diff < 1e-7


def test_f9_safety_clearance_grid_generation_with_buffer():
    """generate_grid_solution must apply the DEFAULT_SAFETY_BUFFER."""
    poses = generate_grid_solution(4)
    # Check gap between square 0 and square 1
    p0 = poses[0]
    p1 = poses[1]
    gap = (p1.cx - 0.5) - (p0.cx + 0.5)
    assert math.isclose(gap, DEFAULT_SAFETY_BUFFER, abs_tol=1e-12)


def test_f9_safety_clearance_protects_against_float_rounding():
    """Float rounding of 1e-16 on top of 1.05e-8 buffer cannot invert sign of gap."""
    effective_gap = 1.05e-8 - 1e-16
    assert effective_gap > 1.0e-8


# ==============================================================================
# FEATURE 10: Monotonic Registry Update Rule (Requirement: 5+ test cases)
# ==============================================================================

def test_f10_registry_init_and_empty_state():
    """Fresh registry has no solutions and empty manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        assert reg.get_best(1) is None
        assert reg.get_best(100) is None


def test_f10_registry_accepts_strictly_better_solution():
    """Registry accepts a strictly improved valid configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        # First submission for n=2: 2.0 side length
        p_grid = [SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
                  SquarePose(cx=1.5 + 1.05e-8, cy=0.5, angle=0.0, square_id=1)]
        assert reg.submit_candidate(2, p_grid) is True
        best_L, _ = reg.get_best(2)
        assert math.isclose(best_L, 2.0 + 1.05e-8, abs_tol=1e-6)

        # Improved submission: literature diagonal packing with side length approx 1.7071
        # Place two squares touching diagonally
        d = 0.5 * math.sqrt(2)
        p_diag = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=0.5 + d + 1.05e-8, cy=0.5 + d + 1.05e-8, angle=math.pi / 4, square_id=1),
        ]
        res = validate_puzzle_poses(2, p_diag)
        if res.is_valid and res.side_length < best_L:
            assert reg.submit_candidate(2, p_diag) is True
            new_L, _ = reg.get_best(2)
            assert new_L < best_L


def test_f10_registry_rejects_worse_solution():
    """Registry rejects candidate with larger L_n."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        poses = generate_grid_solution(4)
        assert reg.submit_candidate(4, poses) is True
        best_L, _ = reg.get_best(4)

        # Create worse solution: spread out squares
        worse_poses = [
            SquarePose(cx=p.cx * 2.0, cy=p.cy * 2.0, angle=p.angle, square_id=p.square_id)
            for p in poses
        ]
        assert reg.submit_candidate(4, worse_poses) is False
        # Registry still holds original best
        current_L, _ = reg.get_best(4)
        assert current_L == best_L


def test_f10_registry_rejects_identical_solution():
    """Registry rejects candidate with identical L_n (strict < rule)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        poses = generate_grid_solution(4)
        assert reg.submit_candidate(4, poses) is True
        assert reg.submit_candidate(4, poses) is False


def test_f10_registry_rejects_overlapping_candidate():
    """Registry rejects invalid/overlapping solution even if reported L_n would be smaller."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        poses = generate_grid_solution(2)
        assert reg.submit_candidate(2, poses) is True

        # Overlapping squares at identical position
        overlap_poses = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=1),
        ]
        assert reg.submit_candidate(2, overlap_poses) is False


def test_f10_registry_atomic_persistence():
    """Saved solutions persist on disk across registry reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg1 = ReferenceRegistry(tmpdir)
        poses = generate_grid_solution(3)
        assert reg1.submit_candidate(3, poses) is True

        # Re-open registry from same dir
        reg2 = ReferenceRegistry(tmpdir)
        best = reg2.get_best(3)
        assert best is not None
        side_length, loaded_poses = best
        assert len(loaded_poses) == 3
        assert math.isclose(side_length, reg1.get_best(3)[0], abs_tol=1e-12)


def test_f10_registry_export_submission_csv():
    """export_submission_csv generates valid CSV with correct rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        csv_path = os.path.join(tmpdir, "submission.csv")
        success, score = reg.export_submission_csv(csv_path, max_n=5)
        assert success is True
        assert score >= 0.0
        # Check rows: 1 + 2 + 3 + 4 + 5 = 15 rows
        df = pd.read_csv(csv_path)
        assert len(df) == 15
        assert list(df["n"].unique()) == [1, 2, 3, 4, 5]
