"""Configuration for SADE V2.5 predicted-objective exploitation."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_4.config import SADEV2_4Config


@dataclass(frozen=True)
class SADEV2_5Config(SADEV2_4Config):
    """V2.4 settings with hard-feasible minimum-mean acquisition."""

