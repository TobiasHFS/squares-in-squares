"""Unit tests for Milestone 2 constructive and literature packing generators."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from src.generators import (
    GOBEL_CONFIGS,
    BaselineAssembler,
    GobelGenerator,
    GridGenerator,
    LStripAugmenter,
    LiteratureOptimaGenerator,
    PythagoreanGenerator,
    augment_l_strip,
    generate_compact_grid,
    generate_gobel_poses,
)
from src.geometry.corners import SquarePose as GeomSquarePose
from src.geometry.validator import validate_puzzle
from src.registry.registry import SolutionRegistry


def test_grid_generator_rigid_regimes():
    gen = GridGenerator()
    for n in [1, 2, 3, 4, 8, 9, 15, 16, 24, 25, 35, 36]:
        poses = gen.generate(n)
        assert poses is not None
        assert len(poses) == n
        geom_poses = [GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in poses]
        res = validate_puzzle(n, geom_poses, safety_buffer=1.05e-8)
        assert res.is_valid
        assert res.side_length <= math.ceil(math.sqrt(n)) + 1e-6


def test_gobel_generator_all_19_configs():
    gen = GobelGenerator()
    assert len(GOBEL_CONFIGS) == 19
    for n, (a, b) in GOBEL_CONFIGS.items():
        poses = gen.generate(n)
        assert poses is not None
        assert len(poses) == n
        geom_poses = [GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in poses]
        res = validate_puzzle(n, geom_poses, safety_buffer=1.05e-8)
        assert res.is_valid, f"Göbel n={n} failed validation: {res.error_message}"
        assert res.side_length < math.ceil(math.sqrt(n)) - 1e-4


def test_literature_optima_generator_small_n():
    gen = LiteratureOptimaGenerator()
    for n in range(1, 37):
        poses = gen.generate(n)
        assert poses is not None
        assert len(poses) == n
        geom_poses = [GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in poses]
        res = validate_puzzle(n, geom_poses, safety_buffer=1.05e-8)
        assert res.is_valid, f"Literature n={n} failed validation: {res.error_message}"
        if n in {5, 10, 17, 26, 27, 28}:
            assert res.side_length < math.ceil(math.sqrt(n)) - 0.1


def test_l_strip_augmenter_chain():
    gen = LStripAugmenter()
    for n in [10, 17, 26, 37, 50, 65, 82, 101]:
        poses = gen.generate(n)
        assert poses is not None
        assert len(poses) == n
        geom_poses = [GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in poses]
        res = validate_puzzle(n, geom_poses, safety_buffer=1.05e-8)
        assert res.is_valid
        assert res.side_length < math.ceil(math.sqrt(n)) - 0.1


def test_baseline_assembler_small_subset():
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_dir = Path(tmpdir) / "registry"
        csv_path = Path(tmpdir) / "sub.csv"
        reg = SolutionRegistry(str(reg_dir))
        assembler = BaselineAssembler(reg)
        metrics = assembler.assemble(max_n=30, output_csv_path=str(csv_path))
        assert metrics.is_valid_export
        assert metrics.populated_puzzles == 30
        assert metrics.assembled_baseline_score < metrics.initial_ceiling_score - 0.3
