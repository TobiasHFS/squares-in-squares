"""Master Baseline Assembler orchestrating constructive generators, geometric validation,
backward monotonic refinement, SolutionRegistry ingestion, and master CSV export.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.generators.base import BaseGenerator
from src.generators.gobel import GobelGenerator
from src.generators.grid import GridGenerator
from src.generators.l_strip import LStripAugmenter
from src.generators.literature_optima import LiteratureOptimaGenerator
from src.generators.pythagorean import PythagoreanGenerator
from src.geometry.corners import SquarePose as GeomSquarePose
from src.geometry.validator import ValidationResult, validate_puzzle, validate_submission_csv
from src.registry.registry import SolutionRegistry
from src.registry.schema import SquarePose

DEFAULT_SAFETY_BUFFER: float = 1.05e-8


@dataclass
class AssemblyMetrics:
    """Summary of baseline assembly execution and verification."""

    total_puzzles: int
    populated_puzzles: int
    initial_ceiling_score: float
    assembled_baseline_score: float
    score_reduction: float
    generator_distribution: Dict[str, int]
    assembly_time_seconds: float
    ingestion_time_seconds: float
    csv_export_time_seconds: float
    csv_validation_time_seconds: float
    csv_file_size_mb: float
    is_valid_export: bool


class BaselineAssembler:
    """Assembles the 100% valid baseline submission across all 1,000 puzzles."""

    def __init__(
        self,
        registry: SolutionRegistry,
        safety_buffer: float = DEFAULT_SAFETY_BUFFER,
        generators: Optional[List[BaseGenerator]] = None,
    ):
        self.registry = registry
        self.safety_buffer = safety_buffer
        self.generators = generators or [
            LiteratureOptimaGenerator(),
            GobelGenerator(),
            LStripAugmenter(),
            GridGenerator(margin=max(1.1e-8, safety_buffer * 1.05)),
        ]

    def assemble(
        self,
        max_n: int = 1000,
        enable_backward_refinement: bool = True,
        output_csv_path: str = "submission.csv",
    ) -> AssemblyMetrics:
        """Executes full assembly pipeline: generation, validation, ingestion, and CSV export."""
        t_start = time.perf_counter()

        # Step 1: Forward multi-candidate generation and validation
        best_candidates: Dict[int, Tuple[float, List[SquarePose], str]] = {}
        for n in range(1, max_n + 1):
            grid_ceiling = float(math.ceil(math.sqrt(n)))
            current_best_L = grid_ceiling + 1e-6
            current_best_poses: Optional[List[SquarePose]] = None
            current_best_gen = "none"

            for gen in self.generators:
                candidate = gen.generate(n)
                if candidate is None or len(candidate) != n:
                    continue

                geom_poses = [
                    GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle) for p in candidate
                ]
                vres: ValidationResult = validate_puzzle(
                    n, geom_poses, safety_buffer=self.safety_buffer
                )

                if vres.is_valid and len(vres.overlapping_pairs) == 0:
                    if vres.side_length < current_best_L - 1e-12:
                        current_best_L = vres.side_length
                        normalized = [
                            SquarePose(cx=p.cx, cy=p.cy, angle=p.angle, square_id=sid)
                            for sid, p in enumerate(candidate)
                        ]
                        current_best_poses = normalized
                        current_best_gen = gen.name

            if current_best_poses is None:
                raise RuntimeError(f"Failed to generate valid baseline configuration for n={n}")

            best_candidates[n] = (current_best_L, current_best_poses, current_best_gen)

        # Step 2: Monotonic backward refinement (subset property)
        if enable_backward_refinement:
            for n in range(max_n - 1, 0, -1):
                next_L, next_poses, next_gen = best_candidates[n + 1]
                curr_L, curr_poses, curr_gen = best_candidates[n]

                if next_L < curr_L - 1e-12:
                    subset_poses = [
                        SquarePose(cx=p.cx, cy=p.cy, angle=p.angle, square_id=sid)
                        for sid, p in enumerate(next_poses[:n])
                    ]
                    geom_subset = [
                        GeomSquarePose(cx=p.cx, cy=p.cy, angle=p.angle)
                        for p in subset_poses
                    ]
                    vres = validate_puzzle(
                        n, geom_subset, safety_buffer=self.safety_buffer
                    )

                    if vres.is_valid and vres.side_length < curr_L - 1e-12:
                        best_candidates[n] = (
                            vres.side_length,
                            subset_poses,
                            f"subset_from_{n+1}({next_gen})",
                        )

        t_assembled = time.perf_counter()

        # Step 3: Registry ingestion
        t_ingest_start = time.perf_counter()
        gen_counts: Dict[str, int] = {}
        for n in range(1, max_n + 1):
            side_len, poses, gen_name = best_candidates[n]
            prefix = gen_name.split("(")[0].split("_")[0]
            gen_counts[prefix] = gen_counts.get(prefix, 0) + 1

            ok = self.registry.submit_candidate(n, poses, generator_name=gen_name)
            if not ok:
                existing = self.registry.get_best(n)
                if existing is None or existing[0] > side_len + 1e-12:
                    raise RuntimeError(f"Registry rejected baseline submission for n={n}")

        t_ingest_end = time.perf_counter()

        # Step 4: CSV Export
        t_export_start = time.perf_counter()
        export_ok, export_score = self.registry.export_submission_csv(
            output_path=output_csv_path,
            max_n=max_n,
            allow_grid_fallback=False,
            verify_roundtrip=True,
        )
        t_export_end = time.perf_counter()

        # Step 5: Full validation round-trip verification
        t_val_start = time.perf_counter()
        all_valid, verified_score, results = validate_submission_csv(
            output_csv_path, expected_puzzles=list(range(1, max_n + 1))
        )
        t_val_end = time.perf_counter()

        ceiling_score = sum(
            (math.ceil(math.sqrt(n)) / math.sqrt(n)) - 1.0 for n in range(1, max_n + 1)
        )
        csv_size_mb = (
            os.path.getsize(output_csv_path) / (1024 * 1024)
            if os.path.exists(output_csv_path)
            else 0.0
        )

        return AssemblyMetrics(
            total_puzzles=max_n,
            populated_puzzles=len(results),
            initial_ceiling_score=ceiling_score,
            assembled_baseline_score=verified_score,
            score_reduction=ceiling_score - verified_score,
            generator_distribution=gen_counts,
            assembly_time_seconds=t_assembled - t_start,
            ingestion_time_seconds=t_ingest_end - t_ingest_start,
            csv_export_time_seconds=t_export_end - t_export_start,
            csv_validation_time_seconds=t_val_end - t_val_start,
            csv_file_size_mb=csv_size_mb,
            is_valid_export=export_ok and all_valid,
        )
