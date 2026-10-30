"""Data models and serialization schemas for the solution registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class SquarePose:
    """Represents the continuous center position and orientation of a unit square."""
    cx: float
    cy: float
    angle: float  # radians counter-clockwise
    square_id: Optional[int] = None
    x_str: str = ""
    y_str: str = ""
    angle_str: str = ""

    def __post_init__(self):
        if not self.x_str:
            cx_norm = 0.0 if self.cx == 0.0 else self.cx
            self.x_str = f"{cx_norm:.17g}"
        if not self.y_str:
            cy_norm = 0.0 if self.cy == 0.0 else self.cy
            self.y_str = f"{cy_norm:.17g}"
        if not self.angle_str:
            angle_norm = 0.0 if self.angle == 0.0 else self.angle
            self.angle_str = f"{angle_norm:.17g}"

    def roundtrip_copy(self) -> SquarePose:
        """Simulates Kaggle float64 CSV write-and-read round-trip using 17g formatting.
        
        Guarantees that coordinates tested by the validator are identical to parsed CSV floats.
        """
        cx_norm = 0.0 if self.cx == 0.0 else self.cx
        cy_norm = 0.0 if self.cy == 0.0 else self.cy
        angle_norm = 0.0 if self.angle == 0.0 else self.angle

        x_str = f"{cx_norm:.17g}"
        y_str = f"{cy_norm:.17g}"
        a_str = f"{angle_norm:.17g}"

        return SquarePose(
            cx=float(x_str),
            cy=float(y_str),
            angle=float(a_str),
            square_id=self.square_id,
            x_str=x_str,
            y_str=y_str,
            angle_str=a_str,
        )

    def to_csv_row(self, row_id: int, n: int, square_id: int) -> str:
        """Formats square pose as a CSV row matching competition specification."""
        cx_norm = 0.0 if self.cx == 0.0 else self.cx
        cy_norm = 0.0 if self.cy == 0.0 else self.cy
        angle_norm = 0.0 if self.angle == 0.0 else self.angle
        return f"{row_id},{n},{square_id},{cx_norm:.17g},{cy_norm:.17g},{angle_norm:.17g}\n"

    def to_dict(self, square_id: Optional[int] = None) -> Dict[str, Any]:
        sid = square_id if square_id is not None else self.square_id
        return {
            "square_id": sid,
            "x": self.cx,
            "y": self.cy,
            "angle": self.angle,
            "x_str": self.x_str,
            "y_str": self.y_str,
            "angle_str": self.angle_str,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SquarePose:
        x_val = float(d.get("x", d.get("cx", 0.0)))
        y_val = float(d.get("y", d.get("cy", 0.0)))
        a_val = float(d.get("angle", 0.0))
        sid = d.get("square_id")
        return cls(
            cx=x_val,
            cy=y_val,
            angle=a_val,
            square_id=int(sid) if sid is not None else None,
            x_str=str(d.get("x_str", f"{x_val:.17g}")),
            y_str=str(d.get("y_str", f"{y_val:.17g}")),
            angle_str=str(d.get("angle_str", f"{a_val:.17g}")),
        )


@dataclass
class SolutionEntry:
    """Persistent solution record for a single puzzle n."""
    n: int
    side_length: float
    score: float
    timestamp: str
    generator: str
    poses: List[SquarePose]
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "side_length": self.side_length,
            "score": self.score,
            "timestamp": self.timestamp,
            "generator": self.generator,
            "iteration": self.iteration,
            "metadata": self.metadata,
            "poses": [p.to_dict(sq_id) for sq_id, p in enumerate(self.poses)],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SolutionEntry:
        poses = [SquarePose.from_dict(p) for p in d.get("poses", [])]
        return cls(
            n=int(d["n"]),
            side_length=float(d["side_length"]),
            score=float(d["score"]),
            timestamp=str(d.get("timestamp", "")),
            generator=str(d.get("generator", "unknown")),
            iteration=int(d.get("iteration", 0)),
            metadata=dict(d.get("metadata", {})),
            poses=poses,
        )


@dataclass
class RegistrySummary:
    """Global summary and score breakdown across all puzzles."""
    total_puzzles: int
    populated_puzzles: int
    total_score: float
    beats_rank1: bool
    target_rank1_score: float
    updated_at: str
    best_scores: Dict[int, float]
    generators: Dict[int, str]
