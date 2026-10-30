"""Public geometric validator and official Kaggle submission scoring suite.

Implements the official metric contracts and float64 CSV serialization round-trip.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

from src.geometry.corners import (
    SquarePose,
    Square,
    compute_corners_float,
    compute_bounding_container_float,
    measure_side_mp,
    HAS_MPMATH,
    L_DPS,
    L_DECIMALS,
)
from src.geometry.sat import verify_puzzle_collision

# Official Kaggle Regex for decimal literals
_DECIMAL_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_MAX_LITERAL_CHARS = 1000
MAX_N = 1000


@dataclass(slots=True)
class ValidationResult:
    """Result of geometric validation for a single puzzle n."""
    is_valid: bool
    side_length: float
    score: float
    error_message: Optional[str] = None
    overlapping_pairs: List[Tuple[int, int]] = field(default_factory=list)


def score_puzzle(n: int, side_length: float) -> float:
    """Official Kaggle score for a feasible packing of n unit squares: L / sqrt(n) - 1.0."""
    if n <= 0:
        return 0.0
    return float(side_length) / math.sqrt(n) - 1.0


def infeasible_score(n: int) -> float:
    """Trivial grid ceiling penalty score: ceil(sqrt(n)) / sqrt(n) - 1.0."""
    if n <= 0:
        return 0.0
    return math.ceil(math.sqrt(n)) / math.sqrt(n) - 1.0


def format_coord_float64(val: float) -> str:
    """Format a float64 literal ensuring exact pandas round-trip without precision loss."""
    # Normalize -0.0 to 0.0
    val_norm = 0.0 if val == 0.0 else val
    return f"{val_norm:.17g}"


def parse_decimal_literal(s: str) -> float:
    """Parse and validate coordinate string literal against official Kaggle regex."""
    if not isinstance(s, str):
        raise ValueError(f"not a string literal: {s!r}")
    s_clean = s.strip()
    if not s_clean or len(s_clean) > _MAX_LITERAL_CHARS or not _DECIMAL_RE.match(s_clean):
        raise ValueError(f"not a decimal number: {s!r}")
    try:
        val = float(s_clean)
    except (ValueError, OverflowError) as e:
        raise ValueError(f"value out of range for a double: {s!r}") from e
    if not math.isfinite(val):
        raise ValueError(f"value out of range for a double: {s!r}")
    return val


def validate_puzzle(
    n: int,
    poses: List[SquarePose],
    safety_buffer: float = 1.05e-8,
) -> ValidationResult:
    """Validates n squares: checks square count, bounds, SAT overlap, and calculates L_n.
    
    Simulates full float64 serialization round-trip to guarantee that the configuration
    survives CSV export and pandas ingestion with zero overlap.
    """
    if n <= 0 or n > MAX_N:
        err = f"Puzzle size n={n} out of valid range [1, {MAX_N}]"
        return ValidationResult(
            is_valid=False,
            side_length=float(math.ceil(math.sqrt(max(1, n)))),
            score=infeasible_score(max(1, n)),
            error_message=err,
            overlapping_pairs=[],
        )

    if len(poses) != n:
        err = f"n={n}: expected {n} poses, got {len(poses)}"
        return ValidationResult(
            is_valid=False,
            side_length=float(math.ceil(math.sqrt(n))),
            score=infeasible_score(n),
            error_message=err,
            overlapping_pairs=[],
        )

    # Convert poses through float64 round-trip serialization
    squares: List[Square] = []
    corners_list: List[List[Tuple[float, float]]] = []

    for sid, pose in enumerate(poses):
        if not (math.isfinite(pose.cx) and math.isfinite(pose.cy) and math.isfinite(pose.angle)):
            err = f"n={n}, square {sid}: non-finite coordinate in pose"
            return ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=err,
                overlapping_pairs=[],
            )

        # Exact decimal string serialization simulation
        x_str = format_coord_float64(pose.cx)
        y_str = format_coord_float64(pose.cy)
        angle_str = format_coord_float64(pose.angle)

        try:
            x_val = parse_decimal_literal(x_str)
            y_val = parse_decimal_literal(y_str)
            angle_val = parse_decimal_literal(angle_str)
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=f"n={n}, square {sid}: {e}",
                overlapping_pairs=[],
            )

        sq = Square(
            x=x_val,
            y=y_val,
            angle=angle_val,
            x_str=x_str,
            y_str=y_str,
            angle_str=angle_str,
        )
        squares.append(sq)
        corners_list.append(sq.corners_float())

    # Execute Collision Verification
    is_feasible, reason, overlapping_pairs = verify_puzzle_collision(
        squares,
        safety_buffer=safety_buffer,
    )

    if not is_feasible:
        return ValidationResult(
            is_valid=False,
            side_length=float(math.ceil(math.sqrt(n))),
            score=infeasible_score(n),
            error_message=f"infeasible: {reason}",
            overlapping_pairs=overlapping_pairs,
        )

    # Compute exact container side length
    side_len = measure_side_mp(squares, dps=L_DPS)
    final_score = score_puzzle(n, side_len)
    return ValidationResult(
        is_valid=True,
        side_length=side_len,
        score=final_score,
        error_message=None,
        overlapping_pairs=[],
    )


def validate_submission_csv(
    csv_path: str,
    expected_puzzles: Optional[List[int]] = None,
) -> Tuple[bool, float, Dict[int, ValidationResult]]:
    """Performs full float64 CSV round-trip and validates all rows against competition rules.
    
    Replicates the official Kaggle score() evaluation backend verbatim.
    
    Returns:
        (all_valid, total_score, results_dict)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission CSV not found: {csv_path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()

    reader = csv.DictReader(io.StringIO(text))
    required_cols = {"n", "square_id", "x", "y", "angle"}
    fieldnames = set(reader.fieldnames or [])
    if not required_cols.issubset(fieldnames):
        raise ValueError(f"Submission CSV missing required columns: {required_cols - fieldnames}")

    puzzles: Dict[int, List[dict]] = {}
    for row_num, row in enumerate(reader, start=2):
        try:
            n = int(row["n"])
        except (ValueError, TypeError):
            raise ValueError(f"Row {row_num}: invalid n value: {row.get('n')!r}")
        if n < 1 or n > MAX_N:
            raise ValueError(f"Row {row_num}: n={n} out of range [1, {MAX_N}]")
        if n not in puzzles:
            puzzles[n] = []
        puzzles[n].append(row)

    target_ns = set(expected_puzzles) if expected_puzzles is not None else set(range(1, MAX_N + 1))
    results: Dict[int, ValidationResult] = {}
    all_valid = True

    for n in sorted(target_ns):
        if n not in puzzles:
            all_valid = False
            results[n] = ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=f"n={n} missing from submission",
                overlapping_pairs=[],
            )
            continue

        rows = puzzles[n]
        if len(rows) != n:
            all_valid = False
            results[n] = ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=f"n={n}: expected {n} rows, got {len(rows)}",
                overlapping_pairs=[],
            )
            continue

        # Check square_id permutation
        try:
            sids = [int(r["square_id"]) for r in rows]
        except (ValueError, TypeError):
            all_valid = False
            results[n] = ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=f"n={n}: invalid square_id format",
                overlapping_pairs=[],
            )
            continue

        if set(sids) != set(range(n)):
            all_valid = False
            results[n] = ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=f"n={n}: square_ids should be 0..{n-1}, got {sorted(sids)}",
                overlapping_pairs=[],
            )
            continue

        # Sort rows by square_id
        rows_by_id = {int(r["square_id"]): r for r in rows}
        poses: List[SquarePose] = []
        parse_error = None
        for sid in range(n):
            r = rows_by_id[sid]
            try:
                cx = parse_decimal_literal(r["x"])
                cy = parse_decimal_literal(r["y"])
                angle = parse_decimal_literal(r["angle"])
                poses.append(SquarePose(cx=cx, cy=cy, angle=angle))
            except ValueError as e:
                parse_error = f"n={n}, square {sid}: {e}"
                break

        if parse_error is not None:
            all_valid = False
            results[n] = ValidationResult(
                is_valid=False,
                side_length=float(math.ceil(math.sqrt(n))),
                score=infeasible_score(n),
                error_message=parse_error,
                overlapping_pairs=[],
            )
            continue

        # Validate with zero additional buffer (matching Kaggle's evaluation exactly)
        vres = validate_puzzle(n, poses, safety_buffer=0.0)
        if not vres.is_valid:
            all_valid = False
        results[n] = vres

    total_score = sum(res.score for res in results.values())
    return all_valid, total_score, results
