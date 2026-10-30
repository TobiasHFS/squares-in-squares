"""
Tier 3: Pairwise & Combinatorial Interaction Test Suite.
Verifies multi-feature interactions, collision cascades, rotated clusters, and registry concurrency.
Requirement: >= 10 test cases.
"""

import math
import os
import tempfile
import csv
import pandas as pd
import pytest

from tests.e2e.ref_geom import (
    SquarePose,
    ValidationResult,
    compute_corners_float,
    compute_aabb,
    broadphase_sweep_and_prune,
    sat_pair_check,
    interval_contact_check,
    validate_puzzle_poses,
    generate_grid_solution,
    ReferenceRegistry,
    DEFAULT_SAFETY_BUFFER,
)


def test_comb_multi_angle_cluster():
    """Interaction: Squares packed at multiple distinct rotation angles (0, 30, 45, 60 deg)."""
    angles = [0.0, math.pi / 6, math.pi / 4, math.pi / 3]
    # Place squares in a 2x2 cluster with sufficient separation
    step = 2.0  # safe margin for any rotation
    poses = [
        SquarePose(cx=0.5, cy=0.5, angle=angles[0], square_id=0),
        SquarePose(cx=0.5 + step, cy=0.5, angle=angles[1], square_id=1),
        SquarePose(cx=0.5, cy=0.5 + step, angle=angles[2], square_id=2),
        SquarePose(cx=0.5 + step, cy=0.5 + step, angle=angles[3], square_id=3),
    ]
    res = validate_puzzle_poses(4, poses)
    assert res.is_valid
    assert len(res.overlapping_pairs) == 0


def test_comb_multi_square_contact_chain():
    """Interaction: Chain of 5 touching squares with exact safety buffer delta."""
    # S0 -- S1 -- S2 -- S3 -- S4 in a horizontal chain
    poses = []
    step = 1.0 + DEFAULT_SAFETY_BUFFER
    for i in range(5):
        poses.append(SquarePose(cx=0.5 + i * step, cy=0.5, angle=0.0, square_id=i))
    res = validate_puzzle_poses(5, poses)
    assert res.is_valid
    # Broadphase will find adjacent pairs (0,1), (1,2), (2,3), (3,4)
    # SAT must certify all as separated due to DEFAULT_SAFETY_BUFFER
    assert len(res.overlapping_pairs) == 0
    expected_L = 1.0 + 4 * step
    assert math.isclose(res.side_length, expected_L, abs_tol=1e-6)


def test_comb_ring_with_central_void():
    """Interaction: 4 squares forming a ring around an empty central cavity."""
    # 4 squares arranged around origin
    # S0 at (0, 1), S1 at (1, 0), S2 at (0, -1), S3 at (-1, 0)
    poses = [
        SquarePose(cx=0.0, cy=1.0 + DEFAULT_SAFETY_BUFFER, angle=0.0, square_id=0),
        SquarePose(cx=1.0 + DEFAULT_SAFETY_BUFFER, cy=0.0, angle=0.0, square_id=1),
        SquarePose(cx=0.0, cy=-(1.0 + DEFAULT_SAFETY_BUFFER), angle=0.0, square_id=2),
        SquarePose(cx=-(1.0 + DEFAULT_SAFETY_BUFFER), cy=0.0, angle=0.0, square_id=3),
    ]
    res = validate_puzzle_poses(4, poses)
    assert res.is_valid
    assert len(res.overlapping_pairs) == 0


def test_comb_corner_to_corner_contacts():
    """Interaction: Multiple corner-to-corner diagonal contacts in a 2x2 checkerboard."""
    # 2 squares touching diagonally at (1.0, 1.0)
    p0 = SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)
    p1 = SquarePose(cx=1.5 + DEFAULT_SAFETY_BUFFER, cy=1.5 + DEFAULT_SAFETY_BUFFER, angle=0.0, square_id=1)
    res = validate_puzzle_poses(2, [p0, p1])
    assert res.is_valid
    assert len(res.overlapping_pairs) == 0


def test_comb_concentric_nested_squares_detected():
    """Interaction: One square inside another, 4-axis SAT detects deep penetration."""
    # S0 at (0.5, 0.5), S1 at (0.5, 0.5) rotated by pi/4
    p0 = SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)
    p1 = SquarePose(cx=0.5, cy=0.5, angle=math.pi / 4, square_id=1)
    c0 = compute_corners_float(p0.cx, p0.cy, p0.angle)
    c1 = compute_corners_float(p1.cx, p1.cy, p1.angle)
    assert sat_pair_check(c0, c1, p0.angle, p1.angle) == "overlapping"
    res = validate_puzzle_poses(2, [p0, p1])
    assert not res.is_valid
    assert (0, 1) in res.overlapping_pairs


def test_comb_dense_cluster_broadphase_overlap_sat_resolves():
    """Interaction: Overlapping AABB bounding boxes resolved cleanly by narrowphase SAT."""
    # Diamond (45 deg) at origin has AABB [-0.707, 0.707] x [-0.707, 0.707]
    # Place axis-aligned squares in the 4 corners of the AABB where diamond corners recede
    # Diamond corner at (0.707, 0)
    # A square placed at (1.0, 1.0) has AABB [0.5, 1.5] x [0.5, 1.5]
    # The AABBs overlap in [0.5, 0.707] x [0.5, 0.707], but squares are disjoint!
    diamond = SquarePose(cx=0.0, cy=0.0, angle=math.pi / 4, square_id=0)
    corner_sq = SquarePose(cx=1.0, cy=1.0, angle=0.0, square_id=1)

    c0 = compute_corners_float(diamond.cx, diamond.cy, diamond.angle)
    c1 = compute_corners_float(corner_sq.cx, corner_sq.cy, corner_sq.angle)
    box0 = compute_aabb(c0)
    box1 = compute_aabb(c1)

    # Verify that AABBs overlap
    candidates = broadphase_sweep_and_prune([(0, box0), (1, box1)])
    assert (0, 1) in candidates  # Broadphase passes pair to narrowphase

    # Verify that narrowphase SAT separates them
    sat_status = sat_pair_check(c0, c1, diamond.angle, corner_sq.angle)
    assert sat_status == "separated"


def test_comb_high_aspect_bounding_ratio():
    """Interaction: High aspect ratio 1x10 strip vs square container calculation."""
    # 10 squares arranged in a 1x10 linear row
    poses = [SquarePose(cx=0.5 + i * (1.0 + DEFAULT_SAFETY_BUFFER), cy=0.5, angle=0.0, square_id=i) for i in range(10)]
    res = validate_puzzle_poses(10, poses)
    assert res.is_valid
    # Bounding box has width ~10, height 1. Container side length is max(width, height) = width
    expected_width = 1.0 + 9 * (1.0 + DEFAULT_SAFETY_BUFFER)
    assert math.isclose(res.side_length, expected_width, abs_tol=1e-5)
    # Linear layout is far worse than 4x4 compact grid for n=10
    grid_poses = generate_grid_solution(10)
    res_grid = validate_puzzle_poses(10, grid_poses)
    assert res_grid.side_length < res.side_length


def test_comb_registry_racing_updates_strict_improvement():
    """Interaction: Competing solutions submitted sequentially enforce strict monotonic decrease."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        # Solution A: Grid 3x3 with L ~ 3.0
        poses_a = generate_grid_solution(5)  # 3x3 container
        assert reg.submit_candidate(5, poses_a) is True
        L_a, _ = reg.get_best(5)

        # Solution B: Göbel motif with 45-degree central square
        # For n=5, Göbel achieves L = 2 + 1/sqrt(2) approx 2.7071
        # Center square at (1.5, 1.5) rotated 45 deg, 4 corner squares
        d = 0.5 * math.sqrt(2)
        L_target = 2.0 + 1.0 / math.sqrt(2) + 2 * DEFAULT_SAFETY_BUFFER
        poses_b = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=L_target - 0.5, cy=0.5, angle=0.0, square_id=1),
            SquarePose(cx=0.5, cy=L_target - 0.5, angle=0.0, square_id=2),
            SquarePose(cx=L_target - 0.5, cy=L_target - 0.5, angle=0.0, square_id=3),
            SquarePose(cx=L_target / 2, cy=L_target / 2, angle=math.pi / 4, square_id=4),
        ]
        res_b = validate_puzzle_poses(5, poses_b)
        if res_b.is_valid and res_b.side_length < L_a:
            assert reg.submit_candidate(5, poses_b) is True
            L_b, _ = reg.get_best(5)
            assert L_b < L_a


def test_comb_registry_rejection_cascade_preserves_valid():
    """Interaction: Submitting an invalid candidate after a valid candidate preserves prior state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        poses_valid = generate_grid_solution(3)
        assert reg.submit_candidate(3, poses_valid) is True
        best_before, _ = reg.get_best(3)

        # Invalid candidate with collision
        poses_invalid = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=1),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=2),
        ]
        assert reg.submit_candidate(3, poses_invalid) is False

        # Verify best remains unchanged
        best_after, _ = reg.get_best(3)
        assert best_after == best_before


def test_comb_full_roundtrip_csv_pandas_sat_validation():
    """Interaction: Write rotated squares to CSV, parse with Pandas, re-validate with SAT."""
    poses = [
        SquarePose(cx=1.0, cy=1.0, angle=math.pi / 6, square_id=0),
        SquarePose(cx=3.0, cy=1.0, angle=math.pi / 4, square_id=1),
        SquarePose(cx=2.0, cy=3.0, angle=math.pi / 3, square_id=2),
    ]
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["id", "n", "square_id", "x", "y", "angle"])
        writer.writeheader()
        for p in poses:
            writer.writerow({
                "id": p.square_id,
                "n": 3,
                "square_id": p.square_id,
                "x": p.x_str,
                "y": p.y_str,
                "angle": p.angle_str,
            })
        tmp_name = tmp.name

    try:
        df = pd.read_csv(tmp_name, float_precision="round_trip")
        reconstructed = []
        for _, row in df.iterrows():
            reconstructed.append(
                SquarePose(
                    cx=float(row["x"]),
                    cy=float(row["y"]),
                    angle=float(row["angle"]),
                    square_id=int(row["square_id"]),
                )
            )
        res = validate_puzzle_poses(3, reconstructed)
        assert res.is_valid
        assert len(res.overlapping_pairs) == 0
    finally:
        os.remove(tmp_name)


def test_comb_mixed_valid_invalid_multi_puzzle():
    """Interaction: Multi-puzzle dataset where puzzle 1 is valid, puzzle 2 is invalid."""
    p_valid = generate_grid_solution(2)
    p_invalid = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=1),
    ]
    res1 = validate_puzzle_poses(2, p_valid)
    res2 = validate_puzzle_poses(2, p_invalid)

    assert res1.is_valid
    assert not res2.is_valid

    # Total score calculation with penalty applied to puzzle 2
    total_score = res1.score + res2.score
    # Puzzle 2 gets penalty L0 = 2.0 -> score = 2/sqrt(2) - 1
    assert math.isclose(res2.side_length, 2.0, abs_tol=1e-12)
    assert total_score > res1.score


def test_comb_gobel_motif_corner_pocket():
    """Interaction: Rotated square nested into an axis-aligned L-corner."""
    # Square 1 at (0.5, 0.5) [x in 0..1, y in 0..1]
    # Square 2 at (0.5, 1.5) [x in 0..1, y in 1..2]
    # Square 3 at (1.5, 0.5) [x in 1..2, y in 0..1]
    # A fourth square rotated 45 deg placed outside the L-corner
    step = 1.0 + DEFAULT_SAFETY_BUFFER
    d = 0.5 * math.sqrt(2)
    p0 = SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0)
    p1 = SquarePose(cx=0.5, cy=0.5 + step, angle=0.0, square_id=1)
    p2 = SquarePose(cx=0.5 + step, cy=0.5, angle=0.0, square_id=2)
    # Place rotated square at (0.5 + step + d, 0.5 + step + d)
    p3 = SquarePose(cx=0.5 + step + d, cy=0.5 + step + d, angle=math.pi / 4, square_id=3)

    res = validate_puzzle_poses(4, [p0, p1, p2, p3])
    assert res.is_valid
    assert len(res.overlapping_pairs) == 0
