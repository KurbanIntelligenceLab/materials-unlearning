"""Validated analysis configuration and repository-relative resource paths."""
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class RepositoryPaths:
    """Resolve resources from a repository root without creating files."""

    root: Path = field(default_factory=Path.cwd)

    @property
    def descriptor_cache(self) -> Path:
        return self.root / "data" / "mp20_full.npz"

    def result(self, *parts: str) -> Path:
        return self.root.joinpath("results", *parts)

    def output(self, analysis: str) -> Path:
        return self.root / "development" / "runs" / analysis


@dataclass(frozen=True)
class RidgeConfiguration:
    """Settings for the exact fixed-feature ridge analysis."""

    features: int = 2048
    regularization: float = 1e-6
    diagnostic_requests: int = 200
    corpus_size: int = 0

    def __post_init__(self):
        if self.features < 1:
            raise ValueError("features must be positive")
        if not isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if self.diagnostic_requests < 3:
            raise ValueError("diagnostic_requests must be at least three")
        if self.corpus_size < 0:
            raise ValueError("corpus_size must be nonnegative")
