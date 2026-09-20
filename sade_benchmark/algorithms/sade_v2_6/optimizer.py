"""SADE V2.6 two-point batch ablation."""

from __future__ import annotations

from ..sade_v2_5.optimizer import SADEV2_5
from ...core.problem import InequalityProblem
from .config import SADEV2_6Config


class SADEV2_6(SADEV2_5):
    """Use one global and one local expensive evaluation per full batch.

    With the V2.5 allocation rule, ``batch_size=2`` and the retained positive
    ``local_fraction`` produce exactly one global slot and one local slot.
    Consequently, the second global boundary-exploration slot is absent while
    every acquisition rule inside the retained slots remains unchanged.
    """

    def __init__(
        self, problem: InequalityProblem, config: SADEV2_6Config | None = None
    ) -> None:
        super().__init__(problem, config or SADEV2_6Config())
        self.config: SADEV2_6Config
