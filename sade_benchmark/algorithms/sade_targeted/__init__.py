"""SADE variant that tightens an existing objective-derived inequality."""

from .config import SADETargetedConfig
from .optimizer import SADETargeted

__all__ = ["SADETargeted", "SADETargetedConfig"]
