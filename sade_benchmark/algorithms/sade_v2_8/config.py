"""Configuration for SADE V2.8 local-neighbor objective reranking."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_7.config import SADEV2_7Config


@dataclass(frozen=True)
class SADEV2_8Config(SADEV2_7Config):
    """V2.7 settings plus a small global-RBF shortlist and 5-NN reranker."""

    objective_shortlist_fraction: float = 0.20
    objective_shortlist_min_size: int = 10
    objective_knn_neighbors: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.objective_shortlist_fraction <= 1.0:
            raise ValueError("objective_shortlist_fraction must be in (0, 1].")
        if self.objective_shortlist_min_size < 1:
            raise ValueError("objective_shortlist_min_size must be positive.")
        if self.objective_knn_neighbors < 1:
            raise ValueError("objective_knn_neighbors must be positive.")
