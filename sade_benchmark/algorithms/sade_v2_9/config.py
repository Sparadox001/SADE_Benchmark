"""Configuration for SADE V2.9."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_7.config import SADEV2_7Config


@dataclass(frozen=True)
class SADEV2_9Config(SADEV2_7Config):
    """V2.7 parameters; V2.9 deliberately introduces no new tuning knob."""
