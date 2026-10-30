"""Unit tests for src/registry/gate.py."""

import math
import pytest
from src.registry.schema import SquarePose
from src.registry.gate import MonotonicGate


def test_gate_rejects_out_of_range_n():
    gate = MonotonicGate()
    acc, entry, reason = gate.evaluate_candidate(0, [], None)
    assert acc is False
    assert "out of valid range" in (reason or "")


def test_gate_rejects_pose_count_mismatch():
    gate = MonotonicGate()
    acc, entry, reason = gate.evaluate_candidate(2, [SquarePose(0.5, 0.5, 0.0)], None)
    assert acc is False
    assert "Expected 2 poses" in (reason or "")


def test_gate_rejects_non_finite_coordinates():
    gate = MonotonicGate()
    poses = [SquarePose(float("inf"), 0.5, 0.0)]
    acc, entry, reason = gate.evaluate_candidate(1, poses, None)
    assert acc is False
    assert "Non-finite float" in (reason or "")


def test_gate_rejects_overlap():
    gate = MonotonicGate()
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(0.7, 0.5, 0.0),
    ]
    acc, entry, reason = gate.evaluate_candidate(2, poses, None)
    assert acc is False
    assert "overlapping" in (reason or "").lower()


def test_gate_accepts_valid_initial_candidate():
    gate = MonotonicGate()
    margin = 1.05e-8
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
    ]
    acc, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=None)
    assert acc is True
    assert entry is not None
    assert entry.n == 2
    assert entry.side_length > 0


def test_gate_rejects_worse_or_equal_candidate():
    gate = MonotonicGate()
    margin = 1.05e-8
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
    ]
    # Current best is 1.9. Candidate side length is ~2.0, so does not improve.
    acc, entry, reason = gate.evaluate_candidate(2, poses, current_best_L=1.9)
    assert acc is False
    assert "does not strictly improve" in (reason or "")


def test_gate_accepts_strictly_better_candidate():
    gate = MonotonicGate()
    # 2x2 grid -> side length 2.000000021
    margin = 1.05e-8
    poses = [
        SquarePose(0.5, 0.5, 0.0),
        SquarePose(1.5 + margin, 0.5, 0.0),
        SquarePose(0.5, 1.5 + margin, 0.0),
        SquarePose(1.5 + margin, 1.5 + margin, 0.0),
    ]
    # Current best is 2.5
    acc, entry, reason = gate.evaluate_candidate(4, poses, current_best_L=2.5)
    assert acc is True
    assert entry is not None
    assert entry.side_length < 2.5
