"""Configuration for SADE V2.12."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_7.config import SADEV2_7Config


@dataclass(frozen=True)
class SADEV2_12Config(SADEV2_7Config):
    """V2.7 parameters; V2.12 introduces no new tuning parameter."""
