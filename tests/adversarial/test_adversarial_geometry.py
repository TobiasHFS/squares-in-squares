"""Adversarial stress testing suite for geometric collision detection and validation.

Adversarially probes:
1. Microscopic overlaps (1e-7 down to 1e-15).
2. Touching boundary contacts (exact zero separation, sub-buffer, and safety buffer).
3. High-degree rotations and irrational angles (Pythagorean, Trump polynomial root).
4. Degenerate collinear edge contacts and vertex-on-vertex contact.
5. Large puzzle sizes (n=1000) with needle-in-a-haystack collision detection.
6. Empirical equivalence and zero false positive / negative rates.
"""

import math
import os
import random
import time
import numpy as np
import pytest

from src.geometry.corners import SquarePose, Square, compute_corners_float
from src.geometry.sat import sat_check_float, sat_check_decimal, verify_puzzle_collision, sort_and_sweep
from src.geometry.validator import validate_puzzle, score_puzzle, infeasible_score
from src.geometry.batch import validate_puzzle_vectorized, closed_form_sat_pairs_numpy
from tests.e2e.ref_geom import sat_pair_check, compute_corners_float as ref_corners


class TestMicroscopicOverlaps:
    """Stress tests probing penetration depths from 1e-7 down to 1e-15."""

    @pytest.mark.parametrize("depth", [1e-7, 1e-8, 1e-9, 1e-12, 1e-15])
    def test_axis_aligned_microscopic_overlaps(self, depth: float):
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 - depth, 0.5, 0.0)]
        
        # Check standard validate_puzzle with default buffer
        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid, f"Failed to reject overlap of depth {depth:.0e} with safety buffer"
        assert res_buf.score == infeasible_score(2)

        # Check standard validate_puzzle with zero buffer (certified interval / decimal check)
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert not res_zero.is_valid, f"Failed to reject overlap of depth {depth:.0e} with safety_buffer=0.0"

        # Check vectorized validator
        cx = np.array([0.5, 1.5 - depth])
        cy = np.array([0.5, 0.5])
        ang = np.array([0.0, 0.0])
        valid, side_len, score, err, ov_pairs = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=1.05e-8)
        assert not valid, f"Vectorized validator failed to reject overlap of depth {depth:.0e}"
        assert len(ov_pairs) > 0

    @pytest.mark.parametrize("depth", [1e-7, 1e-8, 1e-9, 1e-12])
    def test_rotated_microscopic_overlaps_45deg(self, depth: float):
        angle = math.pi / 4.0
        # Normal to edge is (cos 45, sin 45). Overlap along normal by depth:
        D = 1.0 - depth
        cx2 = D * (1.0 / math.sqrt(2.0))
        cy2 = D * (1.0 / math.sqrt(2.0))
        poses = [SquarePose(0.0, 0.0, angle), SquarePose(cx2, cy2, angle)]

        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid, f"Failed to reject 45-deg overlap of depth {depth:.0e}"

        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert not res_zero.is_valid, f"Failed to reject 45-deg overlap with buffer 0 of depth {depth:.0e}"

    @pytest.mark.parametrize("depth", [1e-7, 1e-9])
    def test_rotated_microscopic_overlaps_trump_angle(self, depth: float):
        # Walter Trump's polynomial root angle for n=11 (~40.89 deg)
        theta = 0.7137123456789012
        D = 1.0 - depth
        cx2 = D * math.cos(theta)
        cy2 = D * math.sin(theta)
        poses = [SquarePose(0.0, 0.0, theta), SquarePose(cx2, cy2, theta)]

        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid

        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert not res_zero.is_valid

    @pytest.mark.parametrize("depth", [1e-7, 1e-9, 1e-12])
    def test_vertex_poking_into_edge(self, depth: float):
        # Square 1 at (0.5, 0.5, 0), top edge y=1.0
        # Square 2 at (0.5, 1.0 + 1/sqrt(2) - depth, pi/4) -> bottom vertex penetrates by depth
        cy_base = 1.0 + 0.5 * math.sqrt(2.0)
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(0.5, cy_base - depth, math.pi / 4.0)]

        res = validate_puzzle(2, poses, safety_buffer=0.0)
        assert not res.is_valid, f"Failed to detect vertex poking into edge at depth {depth:.0e}"

    def test_vertex_to_vertex_microscopic_overlap(self):
        # Two diagonal squares touching at (1, 1), overlapping diagonally by 1e-9
        eps = 1e-9
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 - eps, 1.5 - eps, 0.0)]
        res = validate_puzzle(2, poses, safety_buffer=0.0)
        assert not res.is_valid
        assert (0, 1) in res.overlapping_pairs


class TestTouchingBoundaryContacts:
    """Stress tests probing exact zero separation and safety clearance buffer boundaries."""

    def test_exact_zero_separation_axis_aligned(self):
        # Exact edge contact (dx = 1.0, dy = 0.0)
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5, 0.5, 0.0)]

        # Under safety_buffer=0.0 (mathematical non-penetration), it is feasible
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert res_zero.is_valid, "Exact contact should be mathematically feasible with buffer=0.0"

        # Under default safety_buffer=1.05e-8, it must be rejected to prevent float round-trip loss
        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid, "Exact contact must be rejected under safety_buffer=1.05e-8"

    def test_sub_buffer_clearance(self):
        # Gap is 5e-9 (positive separation, but less than safety buffer 1.05e-8)
        # Using a configuration with overlapping AABBs to force SAT evaluation
        angle = math.pi / 4.0
        D = 1.0 + 5e-9
        shift = 0.2
        cx2 = D * (1.0 / math.sqrt(2.0)) - shift * (1.0 / math.sqrt(2.0))
        cy2 = D * (1.0 / math.sqrt(2.0)) + shift * (1.0 / math.sqrt(2.0))
        poses = [SquarePose(0.0, 0.0, angle), SquarePose(cx2, cy2, angle)]

        # Zero buffer should accept
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert res_zero.is_valid

        # Default safety buffer should reject
        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid

    def test_safety_buffer_boundary(self):
        # Gap is 1.05e-8 (the exact required safety buffer)
        angle = math.pi / 4.0
        D = 1.0 + 1.05e-8
        cx2 = D * (1.0 / math.sqrt(2.0))
        cy2 = D * (1.0 / math.sqrt(2.0))
        poses = [SquarePose(0.0, 0.0, angle), SquarePose(cx2, cy2, angle)]

        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert res_buf.is_valid, "Gap 1.05e-8 must be accepted by default safety buffer"

    @pytest.mark.parametrize("gap", [1.02e-8, 1.049e-8])
    def test_sub_buffer_critical_interval_rejection_axis_aligned(self, gap: float):
        """Clearances strictly between 1.00e-8 and 1.05e-8 in axis-aligned configuration
        must be strictly rejected under safety_buffer=1.05e-8 by both validate_puzzle
        and validate_puzzle_vectorized, but accepted under safety_buffer=0.0.
        """
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 + gap, 0.5, 0.0)]
        cx = np.array([0.5, 1.5 + gap])
        cy = np.array([0.5, 0.5])
        ang = np.array([0.0, 0.0])

        # 1. Under default safety_buffer=1.05e-8, validate_puzzle must reject
        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid, f"validate_puzzle erroneously accepted sub-buffer gap {gap:.3e} (axis-aligned)"
        assert res_buf.score == infeasible_score(2)

        # 2. Vectorized validator must also reject
        valid_vec, _, _, _, ov_pairs = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=1.05e-8)
        assert not valid_vec, f"validate_puzzle_vectorized erroneously accepted sub-buffer gap {gap:.3e} (axis-aligned)"
        assert len(ov_pairs) > 0

        # 3. Under safety_buffer=0.0 (exact physical non-penetration), positive clearance is mathematically feasible
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert res_zero.is_valid, f"validate_puzzle rejected physically feasible gap {gap:.3e} with safety_buffer=0.0"

        valid_zero, _, _, _, _ = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=0.0)
        assert valid_zero, f"validate_puzzle_vectorized rejected physically feasible gap {gap:.3e} with safety_buffer=0.0"

    @pytest.mark.parametrize("gap", [1.02e-8, 1.049e-8])
    def test_sub_buffer_critical_interval_rejection_rotated(self, gap: float):
        """Clearances strictly between 1.00e-8 and 1.05e-8 in rotated (pi/4) configuration
        with overlapping AABBs must be strictly rejected under safety_buffer=1.05e-8 by both
        validate_puzzle and validate_puzzle_vectorized, but accepted under safety_buffer=0.0.
        """
        angle = math.pi / 4.0
        D = 1.0 + gap
        shift = 0.2
        cx2 = D * (1.0 / math.sqrt(2.0)) - shift * (1.0 / math.sqrt(2.0))
        cy2 = D * (1.0 / math.sqrt(2.0)) + shift * (1.0 / math.sqrt(2.0))
        poses = [SquarePose(0.0, 0.0, angle), SquarePose(cx2, cy2, angle)]
        cx = np.array([0.0, cx2])
        cy = np.array([0.0, cy2])
        ang = np.array([angle, angle])

        # 1. Under default safety_buffer=1.05e-8, validate_puzzle must reject
        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert not res_buf.is_valid, f"validate_puzzle erroneously accepted sub-buffer gap {gap:.3e} (rotated)"
        assert res_buf.score == infeasible_score(2)

        # 2. Vectorized validator must also reject
        valid_vec, _, _, _, ov_pairs = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=1.05e-8)
        assert not valid_vec, f"validate_puzzle_vectorized erroneously accepted sub-buffer gap {gap:.3e} (rotated)"
        assert len(ov_pairs) > 0

        # 3. Under safety_buffer=0.0, positive clearance is mathematically feasible
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert res_zero.is_valid, f"validate_puzzle rejected physically feasible gap {gap:.3e} with safety_buffer=0.0"

        valid_zero, _, _, _, _ = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=0.0)
        assert valid_zero, f"validate_puzzle_vectorized rejected physically feasible gap {gap:.3e} with safety_buffer=0.0"

    @pytest.mark.parametrize("gap", [1.05e-8, 1.051e-8, 1.06e-8])
    def test_safety_buffer_acceptance_above_threshold_axis_aligned(self, gap: float):
        """Clearances >= 1.05e-8 in axis-aligned configuration must be accepted
        under default safety_buffer=1.05e-8 by both validate_puzzle and validate_puzzle_vectorized.
        """
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 + gap, 0.5, 0.0)]
        cx = np.array([0.5, 1.5 + gap])
        cy = np.array([0.5, 0.5])
        ang = np.array([0.0, 0.0])

        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert res_buf.is_valid, f"validate_puzzle rejected valid gap {gap:.3e} >= 1.05e-8 (axis-aligned)"

        valid_vec, _, _, _, ov_pairs = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=1.05e-8)
        assert valid_vec, f"validate_puzzle_vectorized rejected valid gap {gap:.3e} >= 1.05e-8 (axis-aligned)"
        assert len(ov_pairs) == 0

    @pytest.mark.parametrize("gap", [1.05e-8, 1.051e-8, 1.06e-8])
    def test_safety_buffer_acceptance_above_threshold_rotated(self, gap: float):
        """Clearances >= 1.05e-8 in rotated (pi/4) configuration must be accepted
        under default safety_buffer=1.05e-8 by both validate_puzzle and validate_puzzle_vectorized.
        """
        angle = math.pi / 4.0
        D = 1.0 + gap
        shift = 0.2
        cx2 = D * (1.0 / math.sqrt(2.0)) - shift * (1.0 / math.sqrt(2.0))
        cy2 = D * (1.0 / math.sqrt(2.0)) + shift * (1.0 / math.sqrt(2.0))
        poses = [SquarePose(0.0, 0.0, angle), SquarePose(cx2, cy2, angle)]
        cx = np.array([0.0, cx2])
        cy = np.array([0.0, cy2])
        ang = np.array([angle, angle])

        res_buf = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert res_buf.is_valid, f"validate_puzzle rejected valid gap {gap:.3e} >= 1.05e-8 (rotated)"

        valid_vec, _, _, _, ov_pairs = validate_puzzle_vectorized(2, cx, cy, ang, safety_buffer=1.05e-8)
        assert valid_vec, f"validate_puzzle_vectorized rejected valid gap {gap:.3e} >= 1.05e-8 (rotated)"
        assert len(ov_pairs) == 0

    def test_four_squares_vertex_cluster(self):
        # 4 squares meeting at (1, 1)
        step = 1.0 + 1.05e-8
        poses = [
            SquarePose(0.5, 0.5, 0.0),
            SquarePose(0.5 + step, 0.5, 0.0),
            SquarePose(0.5, 0.5 + step, 0.0),
            SquarePose(0.5 + step, 0.5 + step, 0.0),
        ]
        res = validate_puzzle(4, poses, safety_buffer=1.05e-8)
        assert res.is_valid
        assert math.isclose(res.side_length, 2.0000000105, rel_tol=1e-9)

    def test_diagonal_vertex_on_vertex_touch(self):
        # Only touch at single point (1, 1)
        poses = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5, 1.5, 0.0)]
        res_zero = validate_puzzle(2, poses, safety_buffer=0.0)
        assert res_zero.is_valid


class TestHighDegreeRotationsAndIrrationalAngles:
    """Stress tests on complex rotations, irrational angles, and periodicity."""

    @pytest.mark.parametrize("a,b,c", [(3, 4, 5), (5, 12, 13), (7, 24, 25)])
    def test_pythagorean_rotations(self, a: int, b: int, c: int):
        theta = math.atan2(b, a)
        D = 1.0 + 1.05e-8
        cx2 = D * math.cos(theta)
        cy2 = D * math.sin(theta)
        poses = [SquarePose(0.0, 0.0, theta), SquarePose(cx2, cy2, theta)]

        res = validate_puzzle(2, poses, safety_buffer=1.05e-8)
        assert res.is_valid

        # Slightly overlapping Pythagorean configuration
        poses_ov = [SquarePose(0.0, 0.0, theta), SquarePose((1.0 - 1e-9) * math.cos(theta), (1.0 - 1e-9) * math.sin(theta), theta)]
        res_ov = validate_puzzle(2, poses_ov, safety_buffer=0.0)
        assert not res_ov.is_valid

    def test_trump_angle_sharp_transition(self):
        theta = 0.7137123456789012
        # Exactly 1.05e-8 should pass, 0.99e-8 should fail under buffer=1.05e-8
        poses_pass = [SquarePose(0.0, 0.0, theta), SquarePose((1.0 + 1.05e-8) * math.cos(theta), (1.0 + 1.05e-8) * math.sin(theta), theta)]
        assert validate_puzzle(2, poses_pass, safety_buffer=1.05e-8).is_valid

        poses_fail = [SquarePose(0.0, 0.0, theta), SquarePose((1.0 + 0.99e-8) * math.cos(theta), (1.0 + 0.99e-8) * math.sin(theta), theta)]
        assert not validate_puzzle(2, poses_fail, safety_buffer=1.05e-8).is_valid

    @pytest.mark.parametrize("k", [-1000, -1, 1, 1000])
    def test_angle_periodicity(self, k: int):
        theta = 0.7853981633974483
        theta_k = theta + 2 * math.pi * k
        poses = [SquarePose(0.5, 0.5, theta_k), SquarePose(2.5, 0.5, theta)]
        res = validate_puzzle(2, poses)
        assert res.is_valid


class TestLargeScaleStressWorkloads:
    """Stress tests on large puzzle sizes up to n=1000."""

    def test_compact_grid_n1000_performance(self):
        k = 32
        step = 1.0 + 1.05e-8
        poses = [SquarePose(0.5 + (i % k) * step, 0.5 + (i // k) * step, 0.0) for i in range(1000)]

        t0 = time.perf_counter()
        res = validate_puzzle(1000, poses, safety_buffer=1.05e-8)
        dt = time.perf_counter() - t0

        assert res.is_valid
        assert math.isclose(res.side_length, 32.0, rel_tol=1e-5)
        if os.environ.get("RUN_TIMING_BENCHMARKS") == "1":
            assert dt < 0.20, f"validate_puzzle took {dt:.3f}s, expected < 0.20s"

    def test_needle_in_haystack_overlap_n1000(self):
        k = 32
        step = 1.0 + 1.05e-8
        poses = [SquarePose(0.5 + (i % k) * step, 0.5 + (i // k) * step, 0.0) for i in range(1000)]
        # Inject 1e-9 overlap at (500, 501)
        poses[501] = SquarePose(cx=poses[500].cx + 1.0 - 1e-9, cy=poses[500].cy, angle=0.0)

        res = validate_puzzle(1000, poses, safety_buffer=1.05e-8)
        assert not res.is_valid
        assert (500, 501) in res.overlapping_pairs

        # Also verify with safety_buffer=0.0
        res_zero = validate_puzzle(1000, poses, safety_buffer=0.0)
        assert not res_zero.is_valid
        assert (500, 501) in res_zero.overlapping_pairs

    def test_vectorized_validator_n1000(self):
        k = 32
        step = 1.0 + 1.05e-8
        cx = np.array([0.5 + (i % k) * step for i in range(1000)])
        cy = np.array([0.5 + (i // k) * step for i in range(1000)])
        ang = np.zeros(1000)

        # Valid test
        valid, side_len, score, err, pairs = validate_puzzle_vectorized(1000, cx, cy, ang)
        assert valid
        assert math.isclose(side_len, 32.0, rel_tol=1e-5)

        # Injected overlap
        cx[501] = cx[500] + 1.0 - 1e-9
        valid, side_len, score, err, pairs = validate_puzzle_vectorized(1000, cx, cy, ang)
        assert not valid
        assert (500, 501) in pairs


class TestEquivalenceAndFuzz:
    """Fuzz testing verifying 100% agreement between SAT implementations."""

    def test_fuzz_sat_implementations(self):
        random.seed(1337)
        for i in range(1000):
            x1, y1 = random.uniform(-3, 3), random.uniform(-3, 3)
            # Either completely random or targeted near touching
            if i % 2 == 0:
                dist = random.uniform(0.7, 1.6)
                phi = random.uniform(0, 2 * math.pi)
                x2, y2 = x1 + dist * math.cos(phi), y1 + dist * math.sin(phi)
            else:
                x2, y2 = random.uniform(-3, 3), random.uniform(-3, 3)
            a1 = random.uniform(-math.pi, math.pi)
            a2 = random.uniform(-math.pi, math.pi)

            sq1 = Square(x1, y1, a1)
            sq2 = Square(x2, y2, a2)
            sat_status, _ = sat_check_float(sq1, sq2)

            cx = np.array([x1, x2])
            cy = np.array([y1, y2])
            ang = np.array([a1, a2])
            pairs = np.array([[0, 1]])
            sep_m, ov_m, und_m = closed_form_sat_pairs_numpy(cx, cy, ang, pairs, tol=1e-8)
            batch_status = "separated" if sep_m[0] else ("overlapping" if ov_m[0] else "undecided")

            ref_c1 = ref_corners(x1, y1, a1)
            ref_c2 = ref_corners(x2, y2, a2)
            ref_status = sat_pair_check(ref_c1, ref_c2, a1, a2, tol=1e-8)

            assert sat_status == batch_status == ref_status, (
                f"Disagreement on pair: sat={sat_status}, batch={batch_status}, ref={ref_status}"
            )


class TestMalformedAndBoundaryInputs:
    """Stress tests on non-finite, missing, and malformed inputs."""

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, bad_val: float):
        poses = [SquarePose(bad_val, 0.5, 0.0), SquarePose(1.5, 0.5, 0.0)]
        res = validate_puzzle(2, poses)
        assert not res.is_valid
        assert res.score == infeasible_score(2)

    @pytest.mark.parametrize("bad_n", [0, -1, 1001, 2000])
    def test_out_of_range_n(self, bad_n: int):
        poses = [SquarePose(0.5, 0.5, 0.0)]
        res = validate_puzzle(bad_n, poses)
        assert not res.is_valid

    def test_pose_count_mismatch(self):
        poses = [SquarePose(0.5, 0.5, 0.0)] * 3
        res = validate_puzzle(5, poses)
        assert not res.is_valid
        assert "expected 5 poses, got 3" in res.error_message
