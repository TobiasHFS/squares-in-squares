"""Unit tests for src/registry/registry.py."""

import math
import os
import tempfile
import pandas as pd
import pytest

from src.registry.schema import SquarePose
from src.registry.registry import SolutionRegistry, generate_compact_grid_fallback


def test_compact_grid_fallback():
    poses = generate_compact_grid_fallback(5)
    assert len(poses) == 5
    # ceil(sqrt(5)) = 3
    # Coordinates should be col+0.5, row+0.5 with angle 0
    assert poses[0].cx == 0.5 and poses[0].cy == 0.5
    assert poses[1].cx == 1.5 and poses[1].cy == 0.5
    assert poses[3].cx == 0.5 and poses[3].cy == 1.5


def test_registry_submit_and_get_best():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        assert reg.get_best(1) is None

        # Submit candidate for n=1
        poses_n1 = [SquarePose(0.5, 0.5, 0.0)]
        accepted = reg.submit_candidate(1, poses_n1, generator_name="unit_test")
        assert accepted is True

        best = reg.get_best(1)
        assert best is not None
        side_len, poses = best
        assert math.isclose(side_len, 1.0)
        assert len(poses) == 1

        # Submit candidate for n=2
        margin = 1.05e-8
        poses_n2 = [SquarePose(0.5, 0.5, 0.0), SquarePose(1.5 + margin, 0.5, 0.0)]
        acc2 = reg.submit_candidate(2, poses_n2)
        assert acc2 is True

        # Now submit worse for n=2 (e.g. side length ~2.5) -> must reject
        worse_n2 = [SquarePose(0.5, 0.5, 0.0), SquarePose(2.0, 0.5, 0.0)]
        acc_worse = reg.submit_candidate(2, worse_n2)
        assert acc_worse is False


def test_registry_export_submission_csv_small():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        csv_path = os.path.join(tmpdir, "submission.csv")

        # Export for max_n=3
        success, total_score = reg.export_submission_csv(
            output_path=csv_path, max_n=3, allow_grid_fallback=True
        )
        assert success is True
        assert os.path.isfile(csv_path)

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["id", "n", "square_id", "x", "y", "angle"]
        # Rows: 1 + 2 + 3 = 6 rows
        assert len(df) == 6
        assert list(df["id"]) == list(range(6))


def test_registry_summary_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = SolutionRegistry(registry_dir=tmpdir, auto_load=False)
        summary = reg.get_summary()
        assert summary["total_puzzles"] == 1000
        assert summary["populated_puzzles"] == 0

        # Add solution
        reg.submit_candidate(1, [SquarePose(0.5, 0.5, 0.0)])
        summary2 = reg.get_summary()
        assert summary2["populated_puzzles"] == 1
        assert 1 in summary2["best_scores"] or "1" in summary2["best_scores"]
