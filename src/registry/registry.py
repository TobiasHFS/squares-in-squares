"""Main Solution Registry facade providing persistent storage, monotonic updates,
and master CSV export conforming to PROJECT.md interface contracts.
"""

from __future__ import annotations

import os
import math
import time
import tempfile
import threading
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any, Callable

from src.registry.schema import SquarePose, SolutionEntry, RegistrySummary
from src.registry.storage import ensure_registry_dirs, get_solution_path, atomic_write_json, safe_read_json
from src.registry.gate import MonotonicGate

TARGET_RANK1_SCORE = 22.4160201104


def generate_compact_grid_fallback(n: int) -> List[SquarePose]:
    """Generates the canonical compact grid packing of n unit squares in container of side ceil(sqrt(n))."""
    k = math.ceil(math.sqrt(n))
    poses = []
    for i in range(n):
        col = i % k
        row = i // k
        poses.append(SquarePose(cx=col + 0.5, cy=row + 0.5, angle=0.0, square_id=i))
    return poses


class SolutionRegistry:
    """Central monotonic solution registry for Kaggle 'Squares in Squares'.
    
    Guarantees atomic persistence, strict monotonic improvements, and 500,500-row export.
    """
    def __init__(
        self,
        registry_dir: str = "registry",
        validator: Optional[Callable] = None,
        auto_load: bool = True,
    ):
        self.registry_dir = registry_dir
        self.solutions_dir = ensure_registry_dirs(registry_dir)
        self.summary_path = os.path.join(registry_dir, "summary.json")
        self.gate = MonotonicGate(validator_fn=validator)
        self._lock = threading.RLock()
        self._cache: Dict[int, SolutionEntry] = {}

        if auto_load:
            self._load_cache()

    def _load_cache(self) -> None:
        """Loads all existing solution JSON files into memory cache."""
        for n in range(1, 1001):
            fp = get_solution_path(self.registry_dir, n)
            data = safe_read_json(fp)
            if data is not None:
                try:
                    self._cache[n] = SolutionEntry.from_dict(data)
                except Exception:
                    pass

    def get_best(self, n: int) -> Optional[Tuple[float, List[SquarePose]]]:
        """Returns best (side_length, poses) for puzzle n, or None if not populated."""
        with self._lock:
            entry = self._cache.get(n)
            if entry is not None:
                return entry.side_length, entry.poses
            # Disk check fallback
            fp = get_solution_path(self.registry_dir, n)
            data = safe_read_json(fp)
            if data is not None:
                try:
                    entry = SolutionEntry.from_dict(data)
                    self._cache[n] = entry
                    return entry.side_length, entry.poses
                except Exception:
                    return None
            return None

    def get_best_entry(self, n: int) -> Optional[SolutionEntry]:
        """Returns full SolutionEntry record for puzzle n."""
        with self._lock:
            if n in self._cache:
                return self._cache[n]
            fp = get_solution_path(self.registry_dir, n)
            data = safe_read_json(fp)
            if data is not None:
                try:
                    entry = SolutionEntry.from_dict(data)
                    self._cache[n] = entry
                    return entry
                except Exception:
                    return None
            return None

    def submit_candidate(
        self,
        n: int,
        poses: List[SquarePose],
        generator_name: str = "unknown",
    ) -> bool:
        """Validates candidate configuration and atomically updates registry if strictly improved.
        
        Returns True if accepted and committed, False otherwise.
        """
        with self._lock:
            # Re-check current best (both memory and disk to guard against race conditions)
            current_entry = self.get_best_entry(n)
            current_best_L = current_entry.side_length if current_entry else None

            # Monotonic gate evaluation
            accepted, entry, reason = self.gate.evaluate_candidate(
                n=n,
                poses=poses,
                current_best_L=current_best_L,
                generator_name=generator_name,
            )
            if not accepted or entry is None:
                return False

            # Commit solution atomically to disk
            fp = get_solution_path(self.registry_dir, n)
            atomic_write_json(fp, entry.to_dict())

            # Update cache
            self._cache[n] = entry

            # Update global summary
            self._update_summary_file()
            return True

    def export_submission_csv(
        self,
        output_path: str = "submission.csv",
        max_n: int = 1000,
        allow_grid_fallback: bool = True,
        verify_roundtrip: bool = True,
    ) -> Tuple[bool, float]:
        """Generates master submission.csv containing exactly formatted rows.
        
        Uses atomic file replacement to prevent corrupt or partial output files.
        Returns (success: bool, total_score: float).
        """
        with self._lock:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)

            prefix = f"tmp_{os.path.basename(output_path)}_"
            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                dir=output_dir,
                prefix=prefix,
                suffix=".tmp",
                delete=False,
                newline="",
                encoding="utf-8",
            )
            tmp_csv = tmp.name

            total_score = 0.0
            row_id = 0
            batch_lines = ["id,n,square_id,x,y,angle\n"]

            try:
                for n in range(1, max_n + 1):
                    best = self.get_best(n)
                    if best is not None:
                        side_len, poses = best
                    elif allow_grid_fallback:
                        poses = generate_compact_grid_fallback(n)
                        k = math.ceil(math.sqrt(n))
                        side_len = float(k)
                    else:
                        raise ValueError(f"Puzzle n={n} missing in registry and allow_grid_fallback=False")

                    # Accumulate score
                    score_n = (side_len / math.sqrt(n)) - 1.0
                    total_score += score_n

                    # Write rows for puzzle n
                    for sq_id, p in enumerate(poses):
                        batch_lines.append(p.to_csv_row(row_id, n, sq_id))
                        row_id += 1

                    if len(batch_lines) >= 50000:
                        tmp.writelines(batch_lines)
                        batch_lines.clear()

                if batch_lines:
                    tmp.writelines(batch_lines)
                    batch_lines.clear()

                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()

                # Verification step on isolated tempfile BEFORE atomic replacement
                if verify_roundtrip:
                    expected_rows = (max_n * (max_n + 1)) // 2
                    valid_export = False
                    with open(tmp_csv, "r", encoding="utf-8") as f:
                        header = f.readline().strip()
                        if header == "id,n,square_id,x,y,angle":
                            count = sum(1 for _ in f)
                            if count == expected_rows:
                                valid_export = True

                    if not valid_export:
                        if os.path.exists(tmp_csv):
                            try:
                                os.unlink(tmp_csv)
                            except OSError:
                                pass
                        return False, total_score

                # Atomic replacement with exponential backoff retry loop
                max_retries = 8
                for attempt in range(max_retries):
                    try:
                        os.replace(tmp_csv, output_path)
                        break
                    except (PermissionError, OSError) as e:
                        if attempt == max_retries - 1:
                            raise OSError(f"Failed to replace {output_path}: {e}") from e
                        time.sleep(0.02 * (2 ** attempt))

                return True, total_score
            except Exception:
                try:
                    tmp.close()
                except Exception:
                    pass
                if os.path.exists(tmp_csv):
                    try:
                        os.unlink(tmp_csv)
                    except OSError:
                        pass
                raise

    def get_total_score(self, allow_grid_fallback: bool = True) -> float:
        """Calculates total competition score across all 1,000 puzzles."""
        total = 0.0
        with self._lock:
            for n in range(1, 1001):
                entry = self._cache.get(n)
                if entry is not None:
                    total += (entry.side_length / math.sqrt(n)) - 1.0
                elif allow_grid_fallback:
                    k = math.ceil(math.sqrt(n))
                    total += (float(k) / math.sqrt(n)) - 1.0
        return total

    def _update_summary_file(self) -> None:
        """Writes registry summary to summary.json atomically."""
        total_score = self.get_total_score(allow_grid_fallback=True)
        summary_data = {
            "total_puzzles": 1000,
            "populated_puzzles": len(self._cache),
            "total_score": total_score,
            "target_rank1_score": TARGET_RANK1_SCORE,
            "beats_rank1": total_score < TARGET_RANK1_SCORE,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "best_scores": {n: entry.score for n, entry in self._cache.items()},
            "generators": {n: entry.generator for n, entry in self._cache.items()},
        }
        atomic_write_json(self.summary_path, summary_data)

    def get_summary(self) -> Dict[str, Any]:
        """Returns in-memory or persisted summary dictionary."""
        data = safe_read_json(self.summary_path)
        if data is not None:
            return data
        self._update_summary_file()
        return safe_read_json(self.summary_path) or {}
