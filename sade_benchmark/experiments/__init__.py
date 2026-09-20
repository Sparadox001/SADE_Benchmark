"""Experiment expansion, execution helpers, and result persistence."""

from .configuration import DEFAULT_CONFIGURATION, load_configuration
from .persistence import scientific, save_run, write_json, write_rows
from .registry import (
    ALGORITHMS,
    aggregate_rows,
    build_cases,
    make_algorithm_config,
    make_optimizer,
)
from .statistics import (
    DEFAULT_ALPHA,
    DEFAULT_CONTROL,
    DEFAULT_INFEASIBLE_VALUE,
    paper_statistics_rows,
)

__all__ = [
    "ALGORITHMS",
    "DEFAULT_CONFIGURATION",
    "DEFAULT_ALPHA",
    "DEFAULT_CONTROL",
    "DEFAULT_INFEASIBLE_VALUE",
    "aggregate_rows",
    "build_cases",
    "make_optimizer",
    "make_algorithm_config",
    "load_configuration",
    "paper_statistics_rows",
    "save_run",
    "scientific",
    "write_json",
    "write_rows",
]
