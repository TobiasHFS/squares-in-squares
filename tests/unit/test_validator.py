"""Unit tests for src/geometry/validator.py."""

import math
import os
import tempfile
import pytest

from src.geometry.corners import SquarePose
from src.geometry.validator import (
    ValidationResult,
    score_puzzle,
    infeasible_score,
    format_coord_float64,
    parse_decimal_literal,
    validate_puzzle,
    validate_submission_csv,
)


def test_score_puzzle():
    assert math.isclose(score_puzzle(1, 1.0), 0.0)
    assert math.isclose(score_puzzle(4, 2.0), 0.0)
    assert math.isclose(score_puzzle(2, 2.0), 2.0 / math.sqrt(2) - 1.0)


def test_infeasible_score():
    assert math.isclose(infeasible_score(1), 0.0)
    assert math.isclose(infeasible_score(2), 2.0 / math.sqrt(2) - 1.0)
    assert math.isclose(infeasible_score(5), 3.0 / math.sqrt(5) - 1.0)


def test_format_coord_float64():
    assert format_coord_float64(0.0) == "0"
    assert format_coord_float64(-0.0) == "0"
    val = 1.2071067811865475
    s = format_coord_float64(val)
    assert float(s) == val


def test_parse_decimal_literal():
    assert parse_decimal_literal("1.2345") == 1.2345
    assert parse_decimal_literal("  -0.5  ") == -0.5
    assert parse_decimal_literal("1e-5") == 1e-5
    assert parse_decimal_literal("2.5E+2") == 250.0

    with pytest.raises(ValueError):
        parse_decimal_literal("1 + sqrt(2)")
    with pytest.raises(ValueError):
        parse_decimal_literal("NaN")
    with pytest.raises(ValueError):
        parse_decimal_literal("inf")
    with pytest.raises(ValueError):
        parse_decimal_literal("")


def test_validate_puzzle_valid_single():
    res = validate_puzzle(1, [SquarePose(cx=0.5, cy=0.5, angle=0.0)])
    assert res.is_valid is True
    assert math.isclose(res.side_length, 1.0)
    assert math.isclose(res.score, 0.0)
    assert res.overlapping_pairs == []


def test_validate_puzzle_pose_count_mismatch():
    res = validate_puzzle(2, [SquarePose(cx=0.5, cy=0.5, angle=0.0)])
    assert res.is_valid is False
    assert "expected 2 poses, got 1" in (res.error_message or "")


def test_validate_puzzle_non_finite():
    res = validate_puzzle(1, [SquarePose(cx=float("nan"), cy=0.5, angle=0.0)])
    assert res.is_valid is False
    assert "non-finite" in (res.error_message or "")


def test_validate_puzzle_overlapping():
    poses = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0),
        SquarePose(cx=0.8, cy=0.5, angle=0.0),
    ]
    res = validate_puzzle(2, poses)
    assert res.is_valid is False
    assert len(res.overlapping_pairs) > 0


def test_validate_submission_csv_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "submission.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("id,n,square_id,x,y,angle\n")
            f.write("0,1,0,0.5,0.5,0.0\n")
            f.write("1,2,0,0.5,0.5,0.0\n")
            f.write("2,2,1,1.6,0.5,0.0\n")

        valid, total_score, results = validate_submission_csv(csv_path, expected_puzzles=[1, 2])
        assert valid is True
        assert len(results) == 2
        assert results[1].is_valid is True
        assert results[2].is_valid is True


def test_validate_submission_csv_missing_col():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "broken.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("id,n,x,y,angle\n")  # missing square_id

        with pytest.raises(ValueError, match="missing required columns"):
            validate_submission_csv(csv_path)
