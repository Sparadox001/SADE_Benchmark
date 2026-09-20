"""Configuration for SADE V2.11."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2_7.config import SADEV2_7Config


@dataclass(frozen=True)
class SADEV2_11Config(SADEV2_7Config):
    """V2.7 parameters; V2.11 introduces no new tuning parameter."""
