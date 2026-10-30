"""Monotonic improvement gate with float64 round-trip validation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Callable

from src.geometry.corners import SquarePose as GeomSquarePose
from src.geometry.validator import validate_puzzle, ValidationResult
from src.registry.schema import SquarePose, SolutionEntry


class MonotonicGate:
    """Validates candidates through float64 round-trip, geometric collision verification,
    and strict side-length improvement gating.
    """
    def __init__(self, validator_fn: Optional[Callable] = None, eps_tol: float = 1e-12):
        self.validator_fn = validator_fn or self._default_validator
        self.eps_tol = eps_tol

    @staticmethod
    def _default_validator(n: int, poses: List[SquarePose]) -> Tuple[bool, float, float, List[Tuple[int, int]]]:
        # Adapt SquarePose list to geometry SquarePose
        geom_poses = [GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in poses]
        res: ValidationResult = validate_puzzle(n, geom_poses, safety_buffer=1.05e-8)
        return res.is_valid, res.side_length, res.score, res.overlapping_pairs

    def evaluate_candidate(
        self,
        n: int,
        poses: List[SquarePose],
        current_best_L: Optional[float],
        generator_name: str = "unknown",
    ) -> Tuple[bool, Optional[SolutionEntry], Optional[str]]:
        """Evaluates candidate configuration.
        
        Returns:
            (is_accepted, solution_entry, rejection_reason)
        """
        # Step 1: Input bounds and integrity
        if not (1 <= n <= 1000):
            return False, None, f"Puzzle size n={n} out of valid range [1, 1000]"
        if len(poses) != n:
            return False, None, f"Expected {n} poses, got {len(poses)}"
        for sq_id, p in enumerate(poses):
            if not (math.isfinite(p.cx) and math.isfinite(p.cy) and math.isfinite(p.angle)):
                return False, None, f"Non-finite float in square {sq_id}: ({p.cx}, {p.cy}, {p.angle})"

        # Step 2: Simulate float64 CSV serialization round-trip
        rt_poses = [p.roundtrip_copy() for p in poses]

        # Step 3: Independent geometric validation
        is_valid, L_new, score_new, overlaps = self.validator_fn(n, rt_poses)
        if not is_valid or len(overlaps) > 0:
            return False, None, f"Geometric validation rejected: {len(overlaps)} overlapping square pairs"

        # Step 4: Strict monotonic improvement check
        # Allow small tolerance for intentional safety buffer dilation
        grid_ceiling = float(math.ceil(math.sqrt(n)))
        if current_best_L is None:
            if L_new > grid_ceiling + 1e-6:
                return False, None, f"Candidate L={L_new:.6f} exceeds grid ceiling L_0={grid_ceiling:.6f}"
        else:
            if L_new >= current_best_L - self.eps_tol:
                return False, None, f"Candidate L={L_new:.6f} does not strictly improve current best L={current_best_L:.6f}"

        # Step 5: Construct verified solution entry
        now_utc = datetime.now(timezone.utc).isoformat()
        entry = SolutionEntry(
            n=n,
            side_length=L_new,
            score=score_new,
            timestamp=now_utc,
            generator=generator_name,
            poses=rt_poses,
            metadata={
                "is_valid": True,
                "overlapping_pairs": 0,
                "verified_at": now_utc,
            },
        )
        return True, entry, None
