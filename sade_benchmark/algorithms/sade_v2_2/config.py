"""Configuration for SADE V2.2."""

from __future__ import annotations

from dataclasses import dataclass

from ..sade_v2.config import SADEV2Config


@dataclass(frozen=True)
class SADEV2_2Config(SADEV2Config):
    """V2.2 currently adds no parameters beyond V2.1."""

