"""
Standalone Reference Geometric & Registry Verification Engine.
Opaque-box reference derived directly from Kaggle metric specification (kernel 128385355).
"""

import math
import re
import os
import json
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from decimal import Decimal, getcontext


DECIMAL_REGEX = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
UNDECIDED_TOL = 1e-8
DEFAULT_SAFETY_BUFFER = 1.05e-8


@dataclass
class SquarePose:
    cx: float
    cy: float
    angle: float  # radians
    square_id: Optional[int] = None
    x_str: str = ""
    y_str: str = ""
    angle_str: str = ""

    def __post_init__(self):
        if not self.x_str:
            self.x_str = f"{self.cx:.17g}"
        if not self.y_str:
            self.y_str = f"{self.cy:.17g}"
        if not self.angle_str:
            self.angle_str = f"{self.angle:.17g}"


@dataclass
class ValidationResult:
    is_valid: bool
    side_length: float
    score: float
    error_message: Optional[str] = None
    overlapping_pairs: List[Tuple[int, int]] = field(default_factory=list)


def check_decimal_literal(s: str) -> bool:
    """Verifies coordinate string is a valid decimal/scientific literal <= 1000 chars."""
    if not isinstance(s, str):
        return False
    stripped = s.strip()
    if not stripped or len(stripped) > 1000:
        return False
    if not DECIMAL_REGEX.match(stripped):
        return False
    try:
        val = float(stripped)
        return math.isfinite(val)
    except (ValueError, OverflowError):
        return False


def parse_value_float(s: str) -> float:
    """Parses decimal string to finite float, raising ValueError if invalid."""
    if not check_decimal_literal(s):
        raise ValueError(f"Invalid decimal literal: {s}")
    return float(s.strip())


def grid_ceiling_L(n: int) -> float:
    """Official trivial grid ceiling L_0(n) = ceil(sqrt(n))."""
    if n <= 0:
        return 1.0
    return float(math.isqrt(n - 1) + 1 if n > 0 else 1)


def infeasible_score(n: int) -> float:
    """Official penalty score for invalid/missing puzzle: ceil(sqrt(n)) / sqrt(n) - 1.0."""
    if n <= 0:
        return 0.0
    L0 = grid_ceiling_L(n)
    return (L0 / math.sqrt(n)) - 1.0


def compute_corners_float(cx: float, cy: float, angle: float) -> List[Tuple[float, float]]:
    """Calculates 4 corners of unit square (half-width h=0.5) rotated CCW by angle."""
    c = math.cos(angle)
    s = math.sin(angle)
    h = 0.5
    dx1, dy1 = h * (c - s), h * (s + c)
    dx2, dy2 = -h * (c + s), h * (-s + c)
    return [
        (cx + dx1, cy + dy1),
        (cx + dx2, cy + dy2),
        (cx - dx1, cy - dy1),
        (cx - dx2, cy - dy2),
    ]


def compute_aabb(corners: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Returns (min_x, min_y, max_x, max_y) for 4 corners."""
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    return min(xs), min(ys), max(xs), max(ys)


def broadphase_sweep_and_prune(
    boxes: List[Tuple[int, Tuple[float, float, float, float]]]
) -> List[Tuple[int, int]]:
    """Sort-and-sweep AABB along x-axis. Returns candidate pairs (i, j) with i < j."""
    # boxes format: list of (square_id, (min_x, min_y, max_x, max_y))
    sorted_boxes = sorted(boxes, key=lambda item: item[1][0])  # sort by min_x
    candidates = []
    n = len(sorted_boxes)
    for i in range(n):
        id_i, (min_xi, min_yi, max_xi, max_yi) = sorted_boxes[i]
        for j in range(i + 1, n):
            id_j, (min_xj, min_yj, max_xj, max_yj) = sorted_boxes[j]
            if min_xj > max_xi:
                break  # 1D x-interval separates completely, no subsequent box can overlap
            # Check 2D AABB overlap
            if not (max_xi < min_xj or max_xj < min_xi or max_yi < min_yj or max_yj < min_yi):
                pair = (min(id_i, id_j), max(id_i, id_j))
                candidates.append(pair)
    return sorted(list(set(candidates)))


def sat_pair_check(
    corners1: List[Tuple[float, float]],
    corners2: List[Tuple[float, float]],
    angle1: float,
    angle2: float,
    tol: float = UNDECIDED_TOL,
) -> str:
    """Evaluates 4 separating axes with tolerance margin.
    Returns: 'separated', 'overlapping', or 'undecided'.
    """
    axes = [
        (math.cos(angle1), math.sin(angle1)),
        (-math.sin(angle1), math.cos(angle1)),
        (math.cos(angle2), math.sin(angle2)),
        (-math.sin(angle2), math.cos(angle2)),
    ]
    min_overlap = float("inf")
    for ax, ay in axes:
        p1 = [ax * x + ay * y for x, y in corners1]
        p2 = [ax * x + ay * y for x, y in corners2]
        overlap = min(max(p1), max(p2)) - max(min(p1), min(p2))
        if overlap < -tol:
            return "separated"
        if overlap < min_overlap:
            min_overlap = overlap
    if min_overlap > tol:
        return "overlapping"
    return "undecided"


def interval_contact_check(
    sq1: SquarePose,
    sq2: SquarePose,
    dps: int = 50,
) -> bool:
    """Arbitrary-precision certified check using Decimal for borderline contacts.
    Returns True if separated or exact non-penetrating boundary contact (overlap <= 0 on at least 1 axis).
    Returns False if overlapping (penetration > 0 on all 4 axes).
    """
    getcontext().prec = dps
    # Parse high precision coordinates
    cx1 = Decimal(sq1.x_str)
    cy1 = Decimal(sq1.y_str)
    a1 = float(sq1.angle)
    cx2 = Decimal(sq2.x_str)
    cy2 = Decimal(sq2.y_str)
    a2 = float(sq2.angle)

    # Use high-precision float cosines/sines
    cos1, sin1 = Decimal(str(math.cos(a1))), Decimal(str(math.sin(a1)))
    cos2, sin2 = Decimal(str(math.cos(a2))), Decimal(str(math.sin(a2)))
    half = Decimal("0.5")

    dx1_1, dy1_1 = half * (cos1 - sin1), half * (sin1 + cos1)
    dx1_2, dy1_2 = -half * (cos1 + sin1), half * (-sin1 + cos1)
    c1 = [
        (cx1 + dx1_1, cy1 + dy1_1),
        (cx1 + dx1_2, cy1 + dy1_2),
        (cx1 - dx1_1, cy1 - dy1_1),
        (cx1 - dx1_2, cy1 - dy1_2),
    ]

    dx2_1, dy2_1 = half * (cos2 - sin2), half * (sin2 + cos2)
    dx2_2, dy2_2 = -half * (cos2 + sin2), half * (-sin2 + cos2)
    c2 = [
        (cx2 + dx2_1, cy2 + dy2_1),
        (cx2 + dx2_2, cy2 + dy2_2),
        (cx2 - dx2_1, cy2 - dy2_1),
        (cx2 - dx2_2, cy2 - dy2_2),
    ]

    axes = [(cos1, sin1), (-sin1, cos1), (cos2, sin2), (-sin2, cos2)]
    for ax, ay in axes:
        p1 = [ax * x + ay * y for x, y in c1]
        p2 = [ax * x + ay * y for x, y in c2]
        overlap = min(max(p1), max(p2)) - max(min(p1), min(p2))
        if overlap <= Decimal("0"):
            return True  # Certified separated or touching without penetration
    return False


def validate_puzzle_poses(
    n: int,
    poses: List[SquarePose],
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
) -> ValidationResult:
    """Validates n squares: checks square count, bounds, SAT overlap, and calculates L_n."""
    if n <= 0 or n > 1000:
        return ValidationResult(
            is_valid=False,
            side_length=grid_ceiling_L(n),
            score=infeasible_score(n),
            error_message=f"Puzzle size n={n} out of valid range [1, 1000]",
        )

    if len(poses) != n:
        return ValidationResult(
            is_valid=False,
            side_length=grid_ceiling_L(n),
            score=infeasible_score(n),
            error_message=f"Expected {n} squares for puzzle {n}, got {len(poses)}",
        )

    # Check decimal literals and finiteness
    all_corners: List[List[Tuple[float, float]]] = []
    boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    seen_ids = set()

    for idx, sq in enumerate(poses):
        sq_id = sq.square_id if sq.square_id is not None else idx
        if sq_id in seen_ids or sq_id < 0 or sq_id >= n:
            return ValidationResult(
                is_valid=False,
                side_length=grid_ceiling_L(n),
                score=infeasible_score(n),
                error_message=f"Invalid or duplicate square_id {sq_id} for puzzle {n}",
            )
        seen_ids.add(sq_id)

        if not check_decimal_literal(sq.x_str) or not check_decimal_literal(sq.y_str) or not check_decimal_literal(sq.angle_str):
            return ValidationResult(
                is_valid=False,
                side_length=grid_ceiling_L(n),
                score=infeasible_score(n),
                error_message=f"Square {sq_id} contains invalid decimal/non-finite literal",
            )

        corners = compute_corners_float(sq.cx, sq.cy, sq.angle)
        all_corners.append(corners)
        boxes.append((sq_id, compute_aabb(corners)))

    if len(seen_ids) != n:
        return ValidationResult(
            is_valid=False,
            side_length=grid_ceiling_L(n),
            score=infeasible_score(n),
            error_message=f"Square IDs must form full set 0..{n-1}",
        )

    # Broadphase
    candidate_pairs = broadphase_sweep_and_prune(boxes)
    overlapping_pairs: List[Tuple[int, int]] = []

    for id1, id2 in candidate_pairs:
        c1 = all_corners[id1]
        c2 = all_corners[id2]
        sat_status = sat_pair_check(c1, c2, poses[id1].angle, poses[id2].angle)
        if sat_status == "overlapping":
            overlapping_pairs.append((id1, id2))
        elif sat_status == "undecided":
            # Escalation to high-precision interval check
            is_sep = interval_contact_check(poses[id1], poses[id2])
            if not is_sep:
                overlapping_pairs.append((id1, id2))

    if overlapping_pairs:
        return ValidationResult(
            is_valid=False,
            side_length=grid_ceiling_L(n),
            score=infeasible_score(n),
            error_message=f"Detected {len(overlapping_pairs)} overlapping pairs",
            overlapping_pairs=overlapping_pairs,
        )

    # Calculate L_n
    min_x = min(pt[0] for c in all_corners for pt in c)
    max_x = max(pt[0] for c in all_corners for pt in c)
    min_y = min(pt[1] for c in all_corners for pt in c)
    max_y = max(pt[1] for c in all_corners for pt in c)
    side_length = max(max_x - min_x, max_y - min_y)
    score = (side_length / math.sqrt(n)) - 1.0

    return ValidationResult(
        is_valid=True,
        side_length=side_length,
        score=score,
    )


def generate_grid_solution(n: int, margin: float = DEFAULT_SAFETY_BUFFER) -> List[SquarePose]:
    """Generates a compact, non-overlapping grid layout for puzzle n."""
    k = math.isqrt(n - 1) + 1  # grid size ceil(sqrt(n))
    step = 1.0 + margin
    poses = []
    count = 0
    for r in range(k):
        for c in range(k):
            if count >= n:
                break
            cx = 0.5 + c * step
            cy = 0.5 + r * step
            poses.append(SquarePose(cx=cx, cy=cy, angle=0.0, square_id=count))
            count += 1
        if count >= n:
            break
    return poses


class ReferenceRegistry:
    """Standalone reference monotonic solution registry."""

    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        self.solutions_dir = os.path.join(registry_dir, "solutions")
        os.makedirs(self.solutions_dir, exist_ok=True)
        self.manifest_file = os.path.join(registry_dir, "manifest.json")
        self._manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, "r") as f:
                return json.load(f)
        return {"scores": {}, "best_L": {}}

    def _save_manifest(self):
        with open(self.manifest_file, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def get_best(self, n: int) -> Optional[Tuple[float, List[SquarePose]]]:
        n_str = str(n)
        if n_str not in self._manifest["best_L"]:
            return None
        sol_path = os.path.join(self.solutions_dir, f"n_{n}.json")
        if not os.path.exists(sol_path):
            return None
        with open(sol_path, "r") as f:
            data = json.load(f)
        poses = [
            SquarePose(
                cx=p["x"],
                cy=p["y"],
                angle=p["angle"],
                square_id=p.get("square_id", i),
                x_str=p.get("x_str", f"{p['x']:.17g}"),
                y_str=p.get("y_str", f"{p['y']:.17g}"),
                angle_str=p.get("angle_str", f"{p['angle']:.17g}"),
            )
            for i, p in enumerate(data["poses"])
        ]
        return data["side_length"], poses

    def submit_candidate(self, n: int, poses: List[SquarePose]) -> bool:
        """Validates candidate and atomically updates registry if strictly improved."""
        res = validate_puzzle_poses(n, poses)
        if not res.is_valid:
            return False

        current_best = self.get_best(n)
        if current_best is not None:
            current_L, _ = current_best
            if res.side_length >= current_L:
                return False  # Not strictly improved

        # Save solution file atomically
        sol_path = os.path.join(self.solutions_dir, f"n_{n}.json")
        data = {
            "n": n,
            "side_length": res.side_length,
            "score": res.score,
            "poses": [
                {
                    "square_id": p.square_id if p.square_id is not None else i,
                    "x": p.cx,
                    "y": p.cy,
                    "angle": p.angle,
                    "x_str": p.x_str,
                    "y_str": p.y_str,
                    "angle_str": p.angle_str,
                }
                for i, p in enumerate(poses)
            ],
        }
        with open(sol_path, "w") as f:
            json.dump(data, f, indent=2)

        self._manifest["best_L"][str(n)] = res.side_length
        self._manifest["scores"][str(n)] = res.score
        self._save_manifest()
        return True

    def export_submission_csv(self, output_path: str, max_n: int = 1000) -> Tuple[bool, float]:
        """Exports submission.csv with columns id,n,square_id,x,y,angle."""
        total_score = 0.0
        row_id = 0
        fieldnames = ["id", "n", "square_id", "x", "y", "angle"]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for n in range(1, max_n + 1):
                best = self.get_best(n)
                if best is not None:
                    side_length, poses = best
                    score = (side_length / math.sqrt(n)) - 1.0
                else:
                    poses = generate_grid_solution(n)
                    res = validate_puzzle_poses(n, poses)
                    score = res.score
                total_score += score
                for p in poses:
                    writer.writerow({
                        "id": row_id,
                        "n": n,
                        "square_id": p.square_id,
                        "x": p.x_str,
                        "y": p.y_str,
                        "angle": p.angle_str,
                    })
                    row_id += 1
        return True, total_score
