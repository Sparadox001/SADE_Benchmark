"""Configuration for the SADE V2.6 two-point batch ablation."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_5.config import SADEV2_5Config


@dataclass(frozen=True)
class SADEV2_6Config(SADEV2_5Config):
    """V2.5 settings with a two-point batch by default."""

    batch_size: int = 2
