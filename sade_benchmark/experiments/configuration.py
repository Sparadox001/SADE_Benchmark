"""Load and validate reproducible JSON experiment configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
    "experiment": {
        "name": None,
        "algorithms": ["sade"],
        "suites": None,
        "problems": ["all"],
        "dimensions": [10, 30],
        "runs": 30,
        "first_seed": 1,
        "output_dir": None,
    },
    "common": {
        "max_evaluations": 300,
        "population_size": 30,
        "constraint_tolerance": 0.0,
    },
    "algorithms": {
        "sade": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "dynamic_constraint_tightening": False,
            "tightening_trigger_feasible_count": None,
            "tightening_keep_ratio": 0.3,
            "tightening_patience": 3,
            "tightening_improvement_tol": 1e-3,
            "tightening_max_tightens": 100,
            "tightening_min_ratio": 1e-3,
        },
        "sade_dynamic": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "dynamic_constraint_tightening": True,
            "tightening_trigger_feasible_count": None,
            "tightening_keep_ratio": 0.3,
            "tightening_patience": 3,
            "tightening_improvement_tol": 1e-3,
            "tightening_max_tightens": 100,
            "tightening_min_ratio": 1e-3,
        },
        "sade_targeted": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "dynamic_constraint_tightening": True,
            "tightening_constraint_index": -1,
            "tightening_trigger_feasible_count": None,
            "tightening_keep_ratio": 0.3,
            "tightening_patience": 3,
            "tightening_improvement_tol": 1e-3,
            "tightening_max_tightens": 100,
            "tightening_min_ratio": 1e-3,
        },
        "sade_v2": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
        },
        "sade_v2_2": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
        },
        "sade_v2_3": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
        },
        "sade_v2_4": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
        },
        "sade_v2_5": {
            "batch_size": 3,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
        },
        "sade_v2_6": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
        },
        "sade_v2_7": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "sade_v2_8": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
            "objective_shortlist_fraction": 0.20,
            "objective_shortlist_min_size": 10,
            "objective_knn_neighbors": 5,
        },
        "sade_v2_9": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "sade_v2_10": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "sade_v2_11": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "sade_v2_12": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "sade_v2_13": {
            "batch_size": 2,
            "trials_per_target": 6,
            "local_fraction": 0.2,
            "surrogate_min_samples": 30,
            "p_best_fraction": 0.2,
            "distance_threshold": 1e-3,
            "near_feasible_fraction": 0.2,
            "constraint_error_quantile": 0.9,
            "constraint_error_window": 30,
            "constraint_error_min_samples": 12,
            "constraint_safety_factor": 1.0,
            "hard_feasible_min_fraction": 0.05,
            "boundary_stagnation_batches": 3,
        },
        "dsi": {"wmax": 10},
        "dsi_dynamic": {
            "wmax": 10,
            "dynamic_constraint_tightening": True,
            "tightening_trigger_feasible_count": None,
            "tightening_keep_ratio": 0.3,
            "tightening_patience": 3,
            "tightening_improvement_tol": 1e-3,
            "tightening_max_tightens": 100,
            "tightening_min_ratio": 1e-3,
        },
    },
    "recording": {
        "save_evaluations": True,
        "save_populations": True,
        "save_candidate_pools": False,
    },
}

_ALLOWED_KEYS = {
    "experiment": {
        "name",
        "algorithms",
        "suites",
        "problems",
        "dimensions",
        "runs",
        "first_seed",
        "output_dir",
    },
    "common": {
        "max_evaluations",
        "population_size",
        "constraint_tolerance",
    },
    "recording": {
        "save_evaluations",
        "save_populations",
        "save_candidate_pools",
    },
}

_ALGORITHM_KEYS = {
    "sade": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "dynamic_constraint_tightening",
        "tightening_trigger_feasible_count",
        "tightening_keep_ratio",
        "tightening_patience",
        "tightening_improvement_tol",
        "tightening_max_tightens",
        "tightening_min_ratio",
    },
    "sade_dynamic": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "dynamic_constraint_tightening",
        "tightening_trigger_feasible_count",
        "tightening_keep_ratio",
        "tightening_patience",
        "tightening_improvement_tol",
        "tightening_max_tightens",
        "tightening_min_ratio",
    },
    "sade_targeted": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "dynamic_constraint_tightening",
        "tightening_constraint_index",
        "tightening_trigger_feasible_count",
        "tightening_keep_ratio",
        "tightening_patience",
        "tightening_improvement_tol",
        "tightening_max_tightens",
        "tightening_min_ratio",
    },
    "sade_v2": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
    },
    "sade_v2_2": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
    },
    "sade_v2_3": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
    },
    "sade_v2_4": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
    },
    "sade_v2_5": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
    },
    "sade_v2_6": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
    },
    "sade_v2_7": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "sade_v2_8": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
        "objective_shortlist_fraction",
        "objective_shortlist_min_size",
        "objective_knn_neighbors",
    },
    "sade_v2_9": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "sade_v2_10": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "sade_v2_11": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "sade_v2_12": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "sade_v2_13": {
        "batch_size",
        "trials_per_target",
        "local_fraction",
        "surrogate_min_samples",
        "p_best_fraction",
        "distance_threshold",
        "near_feasible_fraction",
        "constraint_error_quantile",
        "constraint_error_window",
        "constraint_error_min_samples",
        "constraint_safety_factor",
        "hard_feasible_min_fraction",
        "boundary_stagnation_batches",
    },
    "dsi": {"wmax"},
    "dsi_dynamic": {
        "wmax",
        "dynamic_constraint_tightening",
        "tightening_trigger_feasible_count",
        "tightening_keep_ratio",
        "tightening_patience",
        "tightening_improvement_tol",
        "tightening_max_tightens",
        "tightening_min_ratio",
    },
}


def _check_keys(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys in configuration section {section!r}: "
            + ", ".join(unknown)
        )


def validate_configuration(values: dict[str, Any]) -> None:
    """Reject misspelled sections and parameters before a long experiment."""

    _check_keys("root", values, set(DEFAULT_CONFIGURATION))
    for section, allowed in _ALLOWED_KEYS.items():
        section_values = values.get(section, {})
        if not isinstance(section_values, dict):
            raise ValueError(f"Configuration section {section!r} must be an object.")
        _check_keys(section, section_values, allowed)

    algorithm_values = values.get("algorithms", {})
    if not isinstance(algorithm_values, dict):
        raise ValueError("Configuration section 'algorithms' must be an object.")
    unknown_algorithms = sorted(set(algorithm_values) - set(_ALGORITHM_KEYS))
    if unknown_algorithms:
        raise ValueError(
            "Unknown algorithm configuration sections: "
            + ", ".join(unknown_algorithms)
        )
    for algorithm, parameters in algorithm_values.items():
        if not isinstance(parameters, dict):
            raise ValueError(f"Algorithm configuration {algorithm!r} must be an object.")
        _check_keys(f"algorithms.{algorithm}", parameters, _ALGORITHM_KEYS[algorithm])


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    """Return defaults merged with an optional JSON configuration file."""

    resolved = copy.deepcopy(DEFAULT_CONFIGURATION)
    if path is None:
        return resolved
    try:
        with path.open("r", encoding="utf-8") as handle:
            supplied = json.load(handle)
    except OSError as error:
        raise ValueError(f"Cannot read experiment configuration {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON configuration {path}: {error}") from error
    if not isinstance(supplied, dict):
        raise ValueError("The experiment configuration root must be a JSON object.")
    validate_configuration(supplied)
    _deep_update(resolved, supplied)
    return resolved
