"""Configuration for the focused SADE V2.10 ablation."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_9.config import SADEV2_9Config


@dataclass(frozen=True)
class SADEV2_10Config(SADEV2_9Config):
    """V2.9 settings; the V2.10 local split is fixed by the ablation."""
