"""Solution Registry package initialization.
"""

from src.registry.schema import SquarePose, SolutionEntry, RegistrySummary
from src.registry.storage import atomic_write_json, safe_read_json, ensure_registry_dirs, get_solution_path
from src.registry.gate import MonotonicGate
from src.registry.registry import (
    SolutionRegistry,
    TARGET_RANK1_SCORE,
    generate_compact_grid_fallback,
)

__all__ = [
    "SquarePose",
    "SolutionEntry",
    "RegistrySummary",
    "atomic_write_json",
    "safe_read_json",
    "ensure_registry_dirs",
    "get_solution_path",
    "MonotonicGate",
    "SolutionRegistry",
    "TARGET_RANK1_SCORE",
    "generate_compact_grid_fallback",
]
