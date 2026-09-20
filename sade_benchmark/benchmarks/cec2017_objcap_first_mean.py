"""Objective-cap CEC2017 variants with first-feasible-mean thresholds."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .cec2017_objcap import CEC2017ObjectiveCapProblem


_SPEC_PATH = Path(__file__).with_name("data") / "cec2017_objcap_first_mean_t0.json"


@lru_cache(maxsize=1)
def _thresholds() -> dict[str, dict[int, float]]:
    with _SPEC_PATH.open("r", encoding="utf-8") as handle:
        specification = json.load(handle)
    if specification["version"] != "first_feasible_mean_t0_v1":
        raise ValueError("Unsupported first-feasible-mean objective-cap version.")
    return {
        name: {int(dimension): float(value) for dimension, value in dimensions.items()}
        for name, dimensions in specification["thresholds"].items()
    }


CEC2017_FIRST_MEAN_DIMENSIONS = {
    name: tuple(dimensions) for name, dimensions in _thresholds().items()
}
CEC2017_FIRST_MEAN_PROBLEMS = tuple(CEC2017_FIRST_MEAN_DIMENSIONS)


class CEC2017FirstMeanProblem(CEC2017ObjectiveCapProblem):
    """Use frozen first-feasible-mean T0 without altering the base CEC2017 suite."""

    def __init__(self, name: str, dimension: int):
        super().__init__(
            name,
            dimension,
            _threshold_map=_thresholds(),
            _suite="cec2017_objcap_first_mean",
        )
