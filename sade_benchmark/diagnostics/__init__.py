"""Offline diagnostics that never feed information back to optimizers."""

from .candidate_oracle import analyze_candidate_pool_results
from .selector_replay import replay_candidate_selectors

__all__ = ["analyze_candidate_pool_results", "replay_candidate_selectors"]
