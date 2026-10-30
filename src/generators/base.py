"""Base generator abstract interface.

All constructive and heuristic packing generators implement BaseGenerator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.registry.schema import SquarePose


class BaseGenerator(ABC):
    """Abstract base class for square packing generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this generator."""
        ...

    @abstractmethod
    def generate(self, n: int) -> Optional[List[SquarePose]]:
        """Generate a valid candidate packing of n unit squares.

        Args:
            n: Puzzle size (number of unit squares, 1 <= n <= 1000).

        Returns:
            List of n SquarePose instances if successful, or None if generator
            does not support or fails to pack puzzle size n.
        """
        ...
