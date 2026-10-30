"""Unit tests for src/registry/schema.py."""

import math
import pytest
from src.registry.schema import SquarePose, SolutionEntry, RegistrySummary


def test_square_pose_roundtrip_copy():
    p = SquarePose(cx=1.2071067811865475, cy=0.5, angle=math.pi / 4, square_id=0)
    rt = p.roundtrip_copy()
    assert math.isclose(rt.cx, p.cx, abs_tol=1e-15)
    assert math.isclose(rt.cy, p.cy, abs_tol=1e-15)
    assert math.isclose(rt.angle, p.angle, abs_tol=1e-15)
    assert rt.square_id == 0


def test_square_pose_to_csv_row():
    p = SquarePose(cx=0.5, cy=1.5, angle=0.0)
    row = p.to_csv_row(row_id=42, n=5, square_id=3)
    assert row == "42,5,3,0.5,1.5,0\n"


def test_square_pose_to_dict_and_from_dict():
    p = SquarePose(cx=2.5, cy=-1.25, angle=0.75, square_id=2)
    d = p.to_dict()
    reconstructed = SquarePose.from_dict(d)
    assert reconstructed.cx == p.cx
    assert reconstructed.cy == p.cy
    assert reconstructed.angle == p.angle
    assert reconstructed.square_id == p.square_id


def test_solution_entry_dict_roundtrip():
    poses = [
        SquarePose(cx=0.5, cy=0.5, angle=0.0, square_id=0),
        SquarePose(cx=1.6, cy=0.5, angle=0.0, square_id=1),
    ]
    entry = SolutionEntry(
        n=2,
        side_length=2.1,
        score=0.4849,
        timestamp="2026-10-22T13:00:00Z",
        generator="test_gen",
        poses=poses,
        iteration=1,
        metadata={"note": "unit test"},
    )
    d = entry.to_dict()
    assert d["n"] == 2
    assert len(d["poses"]) == 2

    recon = SolutionEntry.from_dict(d)
    assert recon.n == entry.n
    assert recon.side_length == entry.side_length
    assert recon.generator == entry.generator
    assert len(recon.poses) == 2
    assert recon.poses[1].cx == 1.6
