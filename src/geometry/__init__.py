"""Geometry and validation subsystem for Kaggle 'Squares in Squares'.

Exports corner calculation, SAT collision detection, batch validation,
and official metric validation API.
"""

from src.geometry.corners import (
    SquarePose,
    Square,
    compute_corners_float,
    compute_aabb,
    aabb_overlap,
    compute_bounding_container_float,
    measure_side_mp,
    L_DPS,
    L_DECIMALS,
)
from src.geometry.sat import (
    get_axes_float,
    project_corners_float,
    sat_check_float,
    sat_check_decimal,
    sort_and_sweep,
    verify_puzzle_collision,
)
from src.geometry.batch import (
    closed_form_sat_pairs_numpy,
    validate_puzzle_vectorized,
    validate_batch_puzzles,
)
from src.geometry.validator import (
    ValidationResult,
    validate_puzzle,
    validate_submission_csv,
    score_puzzle,
    infeasible_score,
    format_coord_float64,
    parse_decimal_literal,
)

__all__ = [
    "SquarePose",
    "Square",
    "compute_corners_float",
    "compute_aabb",
    "aabb_overlap",
    "compute_bounding_container_float",
    "measure_side_mp",
    "L_DPS",
    "L_DECIMALS",
    "get_axes_float",
    "project_corners_float",
    "sat_check_float",
    "sat_check_decimal",
    "sort_and_sweep",
    "verify_puzzle_collision",
    "closed_form_sat_pairs_numpy",
    "validate_puzzle_vectorized",
    "validate_batch_puzzles",
    "ValidationResult",
    "validate_puzzle",
    "validate_submission_csv",
    "score_puzzle",
    "infeasible_score",
    "format_coord_float64",
    "parse_decimal_literal",
]
