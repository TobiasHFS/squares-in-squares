"""Unit tests for src/registry/storage.py."""

import os
import json
import tempfile
import pytest

from src.registry.storage import (
    ensure_registry_dirs,
    get_solution_path,
    atomic_write_json,
    safe_read_json,
)


def test_ensure_registry_dirs_and_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        sol_dir = ensure_registry_dirs(tmpdir)
        assert os.path.isdir(sol_dir)
        sol_path = get_solution_path(tmpdir, 42)
        assert sol_path == os.path.join(tmpdir, "solutions", "n_42.json")


def test_atomic_write_and_safe_read_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_data.json")
        data = {"n": 10, "name": "test", "scores": [1.0, 2.0]}

        atomic_write_json(file_path, data)
        loaded = safe_read_json(file_path)
        assert loaded == data

        # Overwrite atomically
        data["name"] = "updated"
        atomic_write_json(file_path, data)
        updated = safe_read_json(file_path)
        assert updated["name"] == "updated"


def test_safe_read_nonexistent_or_invalid():
    assert safe_read_json("nonexistent_path_xyz_123.json") is None

    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{ invalid json")
        assert safe_read_json(corrupt_path) is None
