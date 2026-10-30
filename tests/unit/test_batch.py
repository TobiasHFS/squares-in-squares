"""Unit tests for src/geometry/batch.py."""

import math
import numpy as np
import pytest
from src.geometry.batch import (
    closed_form_sat_pairs_numpy,
    validate_puzzle_vectorized,
    validate_batch_puzzles,
)


def test_closed_form_sat_pairs_numpy_separated():
    cx = np.array([0.5, 2.5])
    cy = np.array([0.5, 0.5])
    angles = np.array([0.0, 0.0])
    pairs = np.array([[0, 1]])

    sep, ov, und = closed_form_sat_pairs_numpy(cx, cy, angles, pairs, tol=1e-8)
    assert sep[0] is np.True_
    assert ov[0] is np.False_
    assert und[0] is np.False_


def test_closed_form_sat_pairs_numpy_overlapping():
    cx = np.array([0.5, 0.8])
    cy = np.array([0.5, 0.5])
    angles = np.array([0.0, 0.0])
    pairs = np.array([[0, 1]])

    sep, ov, und = closed_form_sat_pairs_numpy(cx, cy, angles, pairs, tol=1e-8)
    assert sep[0] is np.False_
    assert ov[0] is np.True_
    assert und[0] is np.False_


def test_validate_puzzle_vectorized_n1():
    cx = np.array([0.5])
    cy = np.array([0.5])
    angles = np.array([0.0])
    valid, side_len, score, err, ov = validate_puzzle_vectorized(1, cx, cy, angles)
    assert valid is True
    assert math.isclose(side_len, 1.0)
    assert math.isclose(score, 0.0)


def test_validate_puzzle_vectorized_grid4():
    # 2x2 grid with margin 1.05e-8
    margin = 1.05e-8
    cx = np.array([0.5, 1.5 + margin, 0.5, 1.5 + margin])
    cy = np.array([0.5, 0.5, 1.5 + margin, 1.5 + margin])
    angles = np.zeros(4)
    valid, side_len, score, err, ov = validate_puzzle_vectorized(4, cx, cy, angles)
    assert valid is True
    assert side_len > 2.0
    assert math.isclose(side_len, 2.0 + margin, abs_tol=1e-12)
    assert ov == []


def test_validate_puzzle_vectorized_detects_overlap():
    cx = np.array([0.5, 0.7])
    cy = np.array([0.5, 0.5])
    angles = np.zeros(2)
    valid, side_len, score, err, ov = validate_puzzle_vectorized(2, cx, cy, angles)
    assert valid is False
    assert (0, 1) in ov


def test_validate_batch_puzzles():
    puzzles = []
    for n in [1, 2, 4]:
        k = math.ceil(math.sqrt(n))
        step = 1.0 + 1.05e-8
        xs, ys = [], []
        count = 0
        for r in range(k):
            for c in range(k):
                if count >= n:
                    break
                xs.append(0.5 + c * step)
                ys.append(0.5 + r * step)
                count += 1
            if count >= n:
                break
        puzzles.append((n, np.array(xs), np.array(ys), np.zeros(n)))

    all_valid, total_score, results = validate_batch_puzzles(puzzles)
    assert all_valid is True
    assert len(results) == 3
    assert results[1]["is_valid"] is True
    assert results[4]["is_valid"] is True
