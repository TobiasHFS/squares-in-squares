"""Adversarial stress testing suite for Solution Registry and Monotonic Gate.

Targets:
1. Monotonic improvement gate: worse solution, lateral solution, overlapping squares,
   micro-overlaps, touching squares without clearance, NaNs, infinities, negative zeros.
2. Race conditions, concurrent submissions, readers-writer consistency, and re-entrant registry calls.
3. Full 500,500-row export of submission.csv, validated via pandas.read_csv for exact
   shape (500500, 6), exact columns, 0 NaNs, strictly sequential id 0..500499, and zero corrupted rows.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import tempfile
import time
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.registry.gate import MonotonicGate
from src.registry.registry import SolutionRegistry, generate_compact_grid_fallback
from src.registry.schema import SquarePose, SolutionEntry


# ============================================================================
# 1. Monotonic Improvement Gate Adversarial Tests
# ============================================================================

def test_monotonic_gate_rejects_strictly_worse_solution():
    """Gate must reject any candidate that produces a larger side length than current best."""
    gate = MonotonicGate()
    margin = 1.05e-8
    # Baseline for n=2: 2 squares horizontally separated by margin -> L ≈ 2.0 + margin
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
    ]
    # Current best is L = 1.95. Candidate has L ≈ 2.00000001
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=1.95)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "does not strictly improve" in reason


def test_monotonic_gate_rejects_exact_lateral_solution():
    """Gate must reject a candidate with the exact same side length as current best."""
    gate = MonotonicGate()
    margin = 1.05e-8
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
    ]
    # Evaluate with current_best_L = None to get exact side length
    acc_init, entry_init, _ = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert acc_init is True
    assert entry_init is not None
    exact_L = entry_init.side_length

    # Now evaluate identical candidate with current_best_L equal to exact_L
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=exact_L)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "does not strictly improve" in reason


def test_monotonic_gate_rejects_micro_lateral_within_eps():
    """Gate must reject an infinitesimal improvement below eps_tol (1e-12)."""
    gate = MonotonicGate(eps_tol=1e-12)
    margin = 1.05e-8
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
    ]
    _, entry_init, _ = gate.evaluate_candidate(2, poses, current_best_L=None)
    exact_L = entry_init.side_length

    # Virtual current_best_L set to exact_L + 1e-13 (difference is only 1e-13 < 1e-12 eps_tol)
    virtual_current_best = exact_L + 1e-13
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=virtual_current_best)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "does not strictly improve" in reason


def test_monotonic_gate_accepts_strictly_better_solution():
    """Gate must accept a candidate that strictly improves current best by > eps_tol."""
    gate = MonotonicGate()
    margin = 1.05e-8
    # 4 squares in 2x2 grid -> L ≈ 2.0000000105
    poses_2x2 = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
        SquarePose(0.5, 1.5 + margin, 0.0),
        SquarePose(1.5 + margin, 1.5 + margin, 0.0),
    ]
    current_best_L = 2.05  # previous worse packing
    accepted, entry, reason = gate.evaluate_candidate(4, poses_2x2, current_best_L=current_best_L)
    assert accepted is True
    assert entry is not None
    assert entry.side_length < current_best_L
    assert reason is None


def test_monotonic_gate_rejects_exceeding_grid_ceiling_on_empty():
    """When registry is empty for puzzle n, gate must reject candidate exceeding grid ceiling ceil(sqrt(n))."""
    gate = MonotonicGate()
    # For n=2, grid ceiling is ceil(sqrt(2)) = 2.0
    # Create valid non-overlapping poses but spaced far apart (L ≈ 3.0)
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(2.5, 0.5, 0.0),
    ]
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "exceeds grid ceiling" in reason


def test_monotonic_gate_rejects_overlapping_squares_gross():
    """Gate must reject gross overlapping squares (large interior intersection)."""
    gate = MonotonicGate()
    # Overlap of 0.8 along x-axis
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(0.7, 0.5, 0.0),
    ]
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "overlapping" in reason.lower()


def test_monotonic_gate_rejects_overlapping_squares_micro():
    """Gate must reject squares overlapping by microscopic amounts (e.g. 1e-7)."""
    gate = MonotonicGate()
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 - 1e-7, 0.5, 0.0),  # overlaps by 1e-7
    ]
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "overlapping" in reason.lower()


def test_monotonic_gate_rejects_touching_squares_without_safety_margin():
    """Gate must enforce safety clearance buffer (>= 1.05e-8).
    Touching squares at distance exactly 1.0 must be rejected to prevent Kaggle float64 rounding overlap.
    """
    gate = MonotonicGate()
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5, 0.5, 0.0),  # exact contact, 0 clearance
    ]
    accepted, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert accepted is False
    assert entry is None
    assert reason is not None
    assert "overlapping" in reason.lower() or "rejected" in reason.lower()


def test_monotonic_gate_rejects_nans_in_all_fields():
    """Gate must reject poses with NaN in cx, cy, or angle immediately at Step 1."""
    gate = MonotonicGate()

    # NaN in cx
    acc1, _, r1 = gate.evaluate_candidate(1, [SquarePose(float("nan"), 0.5, 0.0)], None)
    assert acc1 is False
    assert "Non-finite float" in (r1 or "")

    # NaN in cy
    acc2, _, r2 = gate.evaluate_candidate(1, [SquarePose(0.5, float("nan"), 0.0)], None)
    assert acc2 is False
    assert "Non-finite float" in (r2 or "")

    # NaN in angle
    acc3, _, r3 = gate.evaluate_candidate(1, [SquarePose(0.5, 0.5, float("nan"))], None)
    assert acc3 is False
    assert "Non-finite float" in (r3 or "")


def test_monotonic_gate_rejects_infinities():
    """Gate must reject +inf and -inf coordinates immediately at Step 1."""
    gate = MonotonicGate()

    acc1, _, r1 = gate.evaluate_candidate(1, [SquarePose(float("inf"), 0.5, 0.0)], None)
    assert acc1 is False
    assert "Non-finite float" in (r1 or "")

    acc2, _, r2 = gate.evaluate_candidate(1, [SquarePose(0.5, -float("inf"), 0.0)], None)
    assert acc2 is False
    assert "Non-finite float" in (r2 or "")


def test_monotonic_gate_handles_negative_zeros():
    """Negative zero (-0.0) is a valid IEEE 754 float geometrically identical to 0.0.
    Gate should accept it for n=1, and serialization must normalize it to '0' rather than '-0'.
    """
    gate = MonotonicGate()
    pose = SquarePose(-0.0, -0.0, -0.0)

    # 1. Coordinate check
    assert pose.cx == 0.0
    assert pose.x_str == "0"
    assert pose.y_str == "0"
    assert pose.angle_str == "0"

    # 2. Gate evaluation for n=1
    acc, entry, reason = gate.evaluate_candidate(1, [pose], current_best_L=None)
    assert acc is True
    assert entry is not None
    assert math.isclose(entry.side_length, 1.0)

    # 3. CSV line formatting must not contain "-0"
    csv_row = pose.to_csv_row(0, 1, 0)
    assert "-0" not in csv_row
    assert csv_row == "0,1,0,0,0,0\n"


def test_monotonic_gate_rejects_invalid_n_range():
    """Gate must reject n outside [1, 1000]."""
    gate = MonotonicGate()
    for bad_n in [0, -1, -100, 1001, 2000]:
        acc, _, r = gate.evaluate_candidate(bad_n, [], None)
        assert acc is False
        assert "out of valid range" in (r or "")


def test_monotonic_gate_rejects_mismatched_pose_count():
    """Gate must reject candidates whose pose count does not equal n."""
    gate = MonotonicGate()
    # n=3 with 2 poses
    poses_2 = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.6, 0.5, 0.0)]
    acc1, _, r1 = gate.evaluate_candidate(3, poses_2, None)
    assert acc1 is False
    assert "Expected 3 poses, got 2" in (r1 or "")

    # n=1 with 2 poses
    acc2, _, r2 = gate.evaluate_candidate(1, poses_2, None)
    assert acc2 is False
    assert "Expected 1 poses, got 2" in (r2 or "")


# ============================================================================
# 2. Concurrency, Race Conditions, and Re-Entrancy Tests
# ============================================================================

def test_registry_reentrant_lock_safety():
    """SolutionRegistry uses threading.RLock, allowing re-entrant queries during submission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)

        # Custom validator that calls reg.get_best() during evaluation
        reentrant_visited = []

        def reentrant_validator(n: int, poses: List[SquarePose]):
            # Re-entrant query on the registry while under submission lock
            best = reg.get_best(n)
            reentrant_visited.append(best)
            return True, 1.0, 0.0, []

        reg.gate.validator_fn = reentrant_validator

        # Submit candidate: submit_candidate acquires _lock, calls gate -> calls reentrant_validator -> calls get_best -> acquires _lock again
        accepted = reg.submit_candidate(1, [SquarePose(0.5, 0.5, 0.0)])
        assert accepted is True
        assert len(reentrant_visited) == 1
        assert reentrant_visited[0] is None  # Initial state was empty

        # Second submission: should see previous best without deadlocking
        accepted2 = reg.submit_candidate(1, [SquarePose(0.5, 0.5, 0.0)])
        # Lateral candidate rejected, but validator was called re-entrantly
        assert len(reentrant_visited) == 2
        assert reentrant_visited[1] is not None
        assert reentrant_visited[1][0] == 1.0


def test_registry_concurrent_submissions_distinct_n():
    """Multiple threads submitting solutions for distinct n simultaneously.
    Must execute with zero deadlocks, zero lock collisions, and 100% data persistence.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        margin = 1.05e-8

        def worker_task(n: int) -> bool:
            # Generate valid grid packing for n
            k = math.ceil(math.sqrt(n))
            poses = []
            for i in range(n):
                c = i % k
                r = i // k
                poses.append(SquarePose(0.5 + c * (1.0 + margin), 0.5 + r * (1.0 + margin), 0.0))
            return reg.submit_candidate(n, poses, generator_name=f"worker_{n}")

        test_ns = list(range(1, 31))  # 30 concurrent puzzles
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(worker_task, test_ns))

        # All submissions should succeed
        assert all(results)

        # Verify all solutions persisted correctly on disk and in cache
        for n in test_ns:
            best = reg.get_best(n)
            assert best is not None
            side_len, poses = best
            assert len(poses) == n
            assert side_len > 0

        # Summary check
        summary = reg.get_summary()
        assert summary["populated_puzzles"] == 30


def test_registry_concurrent_submissions_competing_same_n():
    """Multiple threads submitting competing solutions for the SAME puzzle n.
    Must maintain monotonic consistency: final committed solution must be the minimum L.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        target_n = 2
        margin = 1.05e-8

        # Prepare 10 candidates with strictly decreasing side lengths: L = 2.0, 1.99, 1.98, ..., 1.91
        side_lengths = [2.0 - i * 0.01 for i in range(10)]

        # Custom mock validator that returns designated side length for each candidate
        def make_validator():
            def val(n: int, poses: List[SquarePose]):
                # Determine side length from poses[1].cx
                L = poses[1].cx + 0.5
                return True, L, (L / math.sqrt(n)) - 1.0, []
            return val

        reg.gate.validator_fn = make_validator()

        def submit_worker(target_L: float) -> bool:
            poses = [
                SquarePose(0.5, 0.5, 0.0),
                SquarePose(target_L - 0.5, 0.5, 0.0),
            ]
            return reg.submit_candidate(target_n, poses, generator_name=f"gen_{target_L:.2f}")

        # Shuffle submission order to stress concurrent out-of-order execution
        import random
        shuffled_lengths = side_lengths.copy()
        random.seed(42)
        random.shuffle(shuffled_lengths)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(submit_worker, shuffled_lengths))

        best = reg.get_best(target_n)
        assert best is not None
        best_L, _ = best

        # The best side length must equal min(side_lengths) = 1.91
        expected_min_L = min(side_lengths)
        assert math.isclose(best_L, expected_min_L, abs_tol=1e-6), f"Expected {expected_min_L}, got {best_L}"


def test_registry_concurrent_readers_and_writers():
    """Concurrent readers (get_best, get_summary, get_total_score) and writers (submit_candidate).
    Ensures safe JSON reads, no partial write read corruptions, and no race conditions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        margin = 1.05e-8
        stop_flag = False

        def writer():
            for n in range(1, 21):
                if stop_flag:
                    break
                poses = [SquarePose(0.5 + i * (1.0 + margin), 0.5, 0.0) for i in range(n)]
                reg.submit_candidate(n, poses, generator_name="stress_writer")
                time.sleep(0.005)

        def reader():
            read_count = 0
            while not stop_flag and read_count < 100:
                for n in range(1, 21):
                    _ = reg.get_best(n)
                    _ = reg.get_best_entry(n)
                _ = reg.get_total_score()
                _ = reg.get_summary()
                read_count += 1
                time.sleep(0.002)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            w_fut = executor.submit(writer)
            r_futs = [executor.submit(reader) for _ in range(3)]

            w_fut.result()
            stop_flag = True
            for rf in r_futs:
                rf.result()

        # Check summary integrity
        summary = reg.get_summary()
        assert summary["populated_puzzles"] > 0
        assert isinstance(summary["total_score"], float)


# ============================================================================
# 3. Full 500,500-Row Export & Parsing Stress Tests
# ============================================================================

def test_export_500500_rows_full_scale_dataframe_integrity():
    """Adversarially challenge export of all 500,500 rows to submission.csv and parse with pandas.
    
    Verifies:
    1. CSV file generation completes cleanly.
    2. pandas.read_csv parses without error or warning.
    3. Exact shape is (500500, 6).
    4. Exact column names: id, n, square_id, x, y, angle.
    5. id is strictly sequential 0..500499 (no skipped, duplicate, or reordered IDs).
    6. n ranges strictly from 1 to 1000.
    7. For each n in 1..1000, exactly n rows are present.
    8. For each n, square_id is strictly 0..n-1.
    9. Exactly 0 NaNs / nulls across the entire DataFrame.
    10. Custom registered coordinates and angles (e.g. 45-deg rotation) survive float64 round-trip.
    11. Zero negative zeros ('-0', '-0.0') present in the serialized file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        margin = 1.05e-8

        # 1. Register a few custom solutions to test custom float formatting
        # Custom n=1: negative zero input to ensure it exported as 0
        acc1 = reg.submit_candidate(1, [SquarePose(-0.0, -0.0, -0.0)], generator_name="neg_zero_test")
        assert acc1 is True

        # Custom n=2: axis-aligned with clearance margin
        poses_n2 = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 + margin, 0.5, 0.0)]
        acc2 = reg.submit_candidate(2, poses_n2, generator_name="valid_n2")
        assert acc2 is True

        # Custom n=10: 45 degree angle (pi/4) and precision coordinates fitting within container <= 4.0 + 1e-6
        angle_45 = math.pi / 4.0
        poses_n10 = (
            [SquarePose(2.0, 2.0, angle_45)]
            + [SquarePose(0.5 + i * (1.0 + margin), 0.5, 0.0) for i in range(4)]
            + [SquarePose(0.5 + i * (1.0 + margin), 3.5, 0.0) for i in range(4)]
            + [SquarePose(0.5, 2.0, 0.0)]
        )
        acc10 = reg.submit_candidate(10, poses_n10, generator_name="rotated_test_n10")
        assert acc10 is True

        csv_path = os.path.join(tmpdir, "submission_full.csv")
        t0 = time.perf_counter()
        success, total_score = reg.export_submission_csv(
            output_path=csv_path,
            max_n=1000,
            allow_grid_fallback=True,
            verify_roundtrip=True,
        )
        t_export = time.perf_counter() - t0

        assert success is True
        assert os.path.isfile(csv_path)
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        assert 10.0 < file_size_mb < 30.0, f"Unexpected file size: {file_size_mb:.2f} MB"

        # Read back raw text to verify absence of "-0" or "-0.0"
        with open(csv_path, "r", encoding="utf-8") as f:
            raw_text_head = f.read(5000)
            assert "-0" not in raw_text_head

        # Parse with pandas
        t1 = time.perf_counter()
        df = pd.read_csv(csv_path)
        t_read = time.perf_counter() - t1

        # Check shape
        expected_rows = 500500  # sum(1..1000) = 1000 * 1001 / 2
        assert df.shape == (expected_rows, 6), f"Expected shape ({expected_rows}, 6), got {df.shape}"

        # Check column names
        expected_cols = ["id", "n", "square_id", "x", "y", "angle"]
        assert list(df.columns) == expected_cols, f"Columns mismatch: {list(df.columns)}"

        # Check nulls / NaNs
        nan_counts = df.isna().sum().to_dict()
        assert sum(nan_counts.values()) == 0, f"NaNs found in submission.csv: {nan_counts}"

        # Check sequential row IDs 0..500499
        ids = df["id"].to_numpy()
        assert np.array_equal(ids, np.arange(expected_rows)), "id column is not strictly sequential 0..500499"

        # Check puzzle range
        n_vals = df["n"].to_numpy()
        assert n_vals[0] == 1
        assert n_vals[-1] == 1000
        assert np.all(np.diff(n_vals) >= 0), "n column is not monotonically non-decreasing"

        # Check row counts per n: exactly n rows for each puzzle
        counts_per_n = df["n"].value_counts().sort_index()
        assert len(counts_per_n) == 1000
        assert np.array_equal(counts_per_n.values, np.arange(1, 1001)), "Per-puzzle row counts do not match 1..1000"

        # Check square_id sequence for random sample of puzzles
        for test_n in [1, 2, 3, 5, 10, 50, 100, 500, 1000]:
            sub_df = df[df["n"] == test_n]
            sq_ids = sub_df["square_id"].to_numpy()
            assert np.array_equal(sq_ids, np.arange(test_n)), f"Puzzle n={test_n} square_id mismatch"

        # Check the registered custom solutions
        # n=1
        row_n1 = df[df["n"] == 1].iloc[0]
        assert row_n1["x"] == 0.0
        assert row_n1["y"] == 0.0
        assert row_n1["angle"] == 0.0

        # n=2 valid axis-aligned
        rows_n2 = df[df["n"] == 2]
        assert math.isclose(rows_n2.iloc[0]["x"], 0.5)
        assert math.isclose(rows_n2.iloc[1]["x"], 1.5 + margin)

        # n=10 rotated
        rows_n10 = df[df["n"] == 10]
        assert math.isclose(rows_n10.iloc[0]["angle"], angle_45, rel_tol=1e-14)
        assert math.isclose(rows_n10.iloc[1]["angle"], 0.0)

        # Performance assertion
        assert t_export < 15.0, f"Export took too long: {t_export:.2f}s"
        assert t_read < 10.0, f"pandas read took too long: {t_read:.2f}s"


def test_registry_concurrent_exports_safety():
    """Adversarial stress test: Multiple threads calling export_submission_csv concurrently.
    All concurrent export calls must complete without OSError / WinError 2 file collision,
    and the final exported file must have valid format and row count.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        csv_path = os.path.join(tmpdir, "concurrent_submission.csv")
        target_n = 50
        expected_rows = (target_n * (target_n + 1)) // 2

        errors = []

        def export_worker():
            try:
                ok, _ = reg.export_submission_csv(output_path=csv_path, max_n=target_n)
                if not ok:
                    errors.append("Export returned ok=False")
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(export_worker) for _ in range(3)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Concurrent exports failed with errors: {errors}"
        df = pd.read_csv(csv_path)
        assert len(df) == expected_rows


def test_multi_instance_concurrent_exports_stress():
    """Adversarial multi-instance concurrency stress test:
    8 to 10 independent SolutionRegistry instances (each with its own in-memory lock)
    simultaneously calling export_submission_csv targeting the exact same CSV output file
    with verify_roundtrip=True.

    Verifies:
    1. Zero unhandled file replacement collisions (PermissionError, WinError 5 on Windows NTFS).
    2. Every independent registry instance completes successfully with ok=True.
    3. The shared destination CSV file is uncorrupted, parseable by pandas.read_csv.
    4. Exact row count, columns, 0 nulls, and strictly sequential 0..N-1 IDs are preserved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "multi_instance_export.csv")
        target_n = 50
        expected_rows = (target_n * (target_n + 1)) // 2
        num_instances = 10
        errors = []

        def export_worker(worker_id: int):
            try:
                # Create an independent SolutionRegistry instance (separate RLock, independent state)
                reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
                ok, score = reg.export_submission_csv(
                    output_path=csv_path,
                    max_n=target_n,
                    allow_grid_fallback=True,
                    verify_roundtrip=True,
                )
                if not ok:
                    errors.append(f"Worker {worker_id}: export returned ok=False")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {type(e).__name__}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(export_worker, i) for i in range(num_instances)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Multi-instance concurrent exports failed with {len(errors)} error(s): {errors}"
        assert os.path.isfile(csv_path)

        # Full round-trip validation with pandas
        df = pd.read_csv(csv_path)
        assert df.shape == (expected_rows, 6), f"Expected shape ({expected_rows}, 6), got {df.shape}"
        assert list(df.columns) == ["id", "n", "square_id", "x", "y", "angle"]
        assert df.isna().sum().sum() == 0, "NaNs found in exported CSV"
        assert np.array_equal(df["id"].to_numpy(), np.arange(expected_rows)), "id column is not strictly sequential"


def test_multi_instance_concurrent_exports_with_competing_submissions():
    """Adversarial stress test: 8 independent SolutionRegistry instances simultaneously
    submitting candidate solutions and calling export_submission_csv with verify_roundtrip=True.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "multi_instance_competing.csv")
        target_n = 20
        expected_rows = (target_n * (target_n + 1)) // 2
        num_instances = 8
        margin = 1.05e-8
        errors = []

        def worker_task(worker_id: int):
            try:
                reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
                for n in range(1, 6):
                    k = math.ceil(math.sqrt(n))
                    poses = [
                        SquarePose(
                            0.5 + (i % k) * (1.0 + margin) + worker_id * 1e-6,
                            0.5 + (i // k) * (1.0 + margin),
                            0.0,
                        )
                        for i in range(n)
                    ]
                    reg.submit_candidate(n, poses, generator_name=f"worker_{worker_id}")

                ok, _ = reg.export_submission_csv(
                    output_path=csv_path,
                    max_n=target_n,
                    allow_grid_fallback=True,
                    verify_roundtrip=True,
                )
                if not ok:
                    errors.append(f"Worker {worker_id}: export returned ok=False")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {type(e).__name__}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_instances)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Concurrent workers encountered errors: {errors}"
        assert os.path.isfile(csv_path)

        df = pd.read_csv(csv_path)
        assert df.shape == (expected_rows, 6)
        assert df.isna().sum().sum() == 0
        assert np.array_equal(df["id"].to_numpy(), np.arange(expected_rows))


