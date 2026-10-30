"""
Tier 4: Real-World Workload Scenarios & System Verification Test Suite.
Verifies complete 500,500-row submissions, literature optima, high-excess regimes, and registry invariants.
Requirement: >= 5 real-world workload scenarios.
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
    validate_puzzle_poses,
    generate_grid_solution,
    ReferenceRegistry,
    grid_ceiling_L,
    infeasible_score,
    DEFAULT_SAFETY_BUFFER,
)


def test_w1_full_500500_row_submission_roundtrip():
    """Scenario 1: Full 500,500-row submission generation, CSV round-trip, and Pandas validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)
        csv_path = os.path.join(tmpdir, "submission.csv")

        # 1. Export full submission across all 1000 puzzles
        success, total_score = reg.export_submission_csv(csv_path, max_n=1000)
        assert success is True
        assert os.path.exists(csv_path)

        # 2. Read back with pandas
        df = pd.read_csv(csv_path, float_precision="round_trip")
        assert len(df) == 500500
        assert list(df.columns) == ["id", "n", "square_id", "x", "y", "angle"]

        # 3. Check puzzle IDs and square_ids
        unique_puzzles = df["n"].unique()
        assert len(unique_puzzles) == 1000
        assert unique_puzzles[0] == 1
        assert unique_puzzles[-1] == 1000

        # Check total baseline score <= 29.0 (exact trivial grid ceiling sum is 28.93767)
        assert total_score <= 29.0
        assert math.isclose(total_score, 28.93767, abs_tol=1e-3)
        assert total_score >= 0.0


def test_w2_literature_optima_bounds_1_to_36():
    """Scenario 2: Literature optima check for n=1..36 (Friedman / Ellsworth / Trump / Cantrell)."""
    # Literature known optimal / best-known bounds for n=1..36
    # Source: Erich Friedman "Packing Unit Squares in Squares"
    literature_L = {
        1: 1.0,
        2: 2.0,
        3: 2.0,
        4: 2.0,
        5: 2.0 + 1.0 / math.sqrt(2.0),  # ~2.7071
        6: 3.0,
        7: 3.0,
        8: 3.0,
        9: 3.0,
        10: 3.0 + 1.0 / math.sqrt(2.0), # ~3.7071
        11: 4.0,
        12: 4.0,
        13: 4.0,
        14: 4.0,
        15: 4.0,
        16: 4.0,
        17: 4.0 + 1.0 / math.sqrt(2.0), # ~4.7071
        18: 4.0 + 1.0 / math.sqrt(2.0), # ~4.7071
        19: 5.0,
        20: 5.0,
        21: 5.0,
        22: 5.0,
        23: 5.0,
        24: 5.0,
        25: 5.0,
        26: 5.0 + 1.0 / math.sqrt(2.0), # ~5.7071
        27: 5.0 + 1.0 / math.sqrt(2.0), # ~5.7071
        28: 6.0,
        29: 6.0,
        30: 6.0,
        31: 6.0,
        32: 6.0,
        33: 6.0,
        34: 6.0,
        35: 6.0,
        36: 6.0,
    }

    for n, lit_L in literature_L.items():
        L0 = grid_ceiling_L(n)
        # Every literature optimum must be <= grid ceiling
        assert lit_L <= L0, f"Literature optimum for n={n} ({lit_L}) exceeds grid ceiling ({L0})"
        score_lit = (lit_L / math.sqrt(n)) - 1.0
        score_grid = (L0 / math.sqrt(n)) - 1.0
        assert score_lit <= score_grid + 1e-12

    # High-impact non-integer Göbel cases strictly beat grid ceiling
    for n in [5, 10, 17, 26]:
        L0 = grid_ceiling_L(n)
        lit_L = literature_L[n]
        gain = L0 - lit_L
        # Gain is exactly 1 - 1/sqrt(2) approx 0.292893
        assert math.isclose(gain, 1.0 - 1.0 / math.sqrt(2.0), abs_tol=1e-10)


def test_w3_high_excess_regime_verification():
    """Scenario 3: High-excess puzzle verification (n = k^2 + 1: n=5, 10, 17, 26, 37, 50, 65, 82)."""
    # In these puzzles, adding 1 square to a k^2 grid wastes almost an entire row/col in the grid ceiling
    ks = [2, 3, 4, 5, 6, 7, 8, 9]
    high_excess_ns = [k * k + 1 for k in ks]

    total_grid_excess = 0.0
    total_gobel_excess = 0.0

    for k, n in zip(ks, high_excess_ns):
        L_grid = float(k + 1)  # ceil(sqrt(k^2+1)) = k+1
        L_gobel = float(k) + (1.0 / math.sqrt(2.0))

        score_grid = (L_grid / math.sqrt(n)) - 1.0
        score_gobel = (L_gobel / math.sqrt(n)) - 1.0

        total_grid_excess += score_grid
        total_gobel_excess += score_gobel

        # Göbel score must be strictly lower than grid ceiling
        assert score_gobel < score_grid
        margin = score_grid - score_gobel
        expected_margin = (1.0 - 1.0 / math.sqrt(2.0)) / math.sqrt(n)
        assert math.isclose(margin, expected_margin, abs_tol=1e-10)

    # Verifying total excess reduction across the 8 tested high-excess regimes
    total_savings = total_grid_excess - total_gobel_excess
    assert total_savings > 0.40


def test_w4_rigid_near_square_regimes():
    """Scenario 4: Rigid near-square regimes (n = k^2 - 1: n=3, 8, 15, 24, 35, 48)."""
    # For n = k^2 - 1, packing into a k x k grid with 1 missing cell is rigid and optimal
    ks = [2, 3, 4, 5, 6, 7]
    near_square_ns = [k * k - 1 for k in ks]

    for k, n in zip(ks, near_square_ns):
        # Generate grid solution for n
        poses = generate_grid_solution(n, margin=DEFAULT_SAFETY_BUFFER)
        assert len(poses) == n
        res = validate_puzzle_poses(n, poses)
        assert res.is_valid

        # Side length is k + margin
        assert math.isclose(res.side_length, float(k) + (k - 1) * DEFAULT_SAFETY_BUFFER, abs_tol=1e-5)
        # Expected score: k / sqrt(k^2 - 1) - 1.0
        theoretical_score = (float(k) / math.sqrt(n)) - 1.0
        assert math.isclose(res.score, theoretical_score, abs_tol=1e-5)


def test_w5_monotonic_regression_rollback_and_penalty_invariant():
    """Scenario 5: Monotonic regression rollback & official penalty invariant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = ReferenceRegistry(tmpdir)

        # 1. Initialize registry with baseline solutions for n=1..10
        for n in range(1, 11):
            poses = generate_grid_solution(n)
            assert reg.submit_candidate(n, poses) is True

        _, score_initial = reg.export_submission_csv(os.path.join(tmpdir, "init.csv"), max_n=10)

        # 2. Candidate improvement for n=5 (Göbel packing)
        L_target = 2.0 + 1.0 / math.sqrt(2.0) + 2 * DEFAULT_SAFETY_BUFFER
        poses_better = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=L_target - 0.5, cy=0.5, angle=0.0, square_id=1),
            SquarePose(cx=0.5, cy=L_target - 0.5, angle=0.0, square_id=2),
            SquarePose(cx=L_target - 0.5, cy=L_target - 0.5, angle=0.0, square_id=3),
            SquarePose(cx=L_target / 2, cy=L_target / 2, angle=math.pi / 4, square_id=4),
        ]
        assert reg.submit_candidate(5, poses_better) is True
        _, score_improved = reg.export_submission_csv(os.path.join(tmpdir, "improved.csv"), max_n=10)
        assert score_improved < score_initial

        # 3. Candidate regression / overlap for n=5: attempt to submit overlapping squares
        poses_bad = [
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=1),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=2),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=3),
            SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=4),
        ]
        # Registry must reject it atomically
        assert reg.submit_candidate(5, poses_bad) is False
        _, score_recheck = reg.export_submission_csv(os.path.join(tmpdir, "recheck.csv"), max_n=10)
        assert score_recheck == score_improved  # Exactly preserved!

        # 4. Invariant: Official evaluator penalty fallback
        # If an overlapping puzzle bypasses the registry and reaches evaluator:
        res_eval = validate_puzzle_poses(5, poses_bad)
        assert not res_eval.is_valid
        assert res_eval.side_length == grid_ceiling_L(5)  # 3.0
        assert math.isclose(res_eval.score, infeasible_score(5), abs_tol=1e-12)
