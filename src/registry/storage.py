"""Thread-safe and process-safe atomic persistence layer for Windows NTFS and POSIX.
"""

from __future__ import annotations

import os
import json
import time
import tempfile
from typing import Any, Dict, Optional


def ensure_registry_dirs(registry_dir: str) -> str:
    """Creates registry directory structure if not existing."""
    solutions_dir = os.path.join(registry_dir, "solutions")
    os.makedirs(solutions_dir, exist_ok=True)
    return solutions_dir


def get_solution_path(registry_dir: str, n: int) -> str:
    """Returns absolute path to solution file for puzzle n."""
    return os.path.join(registry_dir, "solutions", f"n_{n}.json")


def atomic_write_json(file_path: str, data: Any, max_retries: int = 5, indent: int = 2) -> None:
    """Atomically writes data to JSON file using a same-directory tempfile, fsync,
    handle closure, and atomic rename with Windows exponential backoff retry.
    """
    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)

    prefix = f"tmp_{os.path.basename(file_path)}_"
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        dir=target_dir,
        prefix=prefix,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = tmp.name
    try:
        json.dump(data, tmp, indent=indent)
        tmp.flush()
        os.fsync(tmp.fileno())
        # Crucial on Windows: handle must be closed before os.replace
        tmp.close()

        # Atomic replace with exponential backoff on transient lock collision
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, file_path)
                break
            except (PermissionError, OSError) as e:
                if attempt == max_retries - 1:
                    raise OSError(
                        f"Failed atomic replacement of {file_path} after {max_retries} attempts: {e}"
                    ) from e
                time.sleep(0.01 * (2 ** attempt))
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def safe_read_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Reads JSON file safely, returning None if file does not exist or read fails."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
