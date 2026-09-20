"""Differential-evolution candidate generation retained from SADE_TED."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...core.problem import FloatArray


MutationStrategy = Callable[[float], FloatArray]


def getF_CR(rng: np.random.Generator) -> tuple[float, float]:
    """Sample the original SADE scale factor F and crossover rate CR."""

    factor = float(np.clip(rng.normal(0.7, 0.3), 0.2, 1.0))
    crossover_rate = float(np.clip(rng.normal(0.5, 0.3), 0.1, 1.0))
    return factor, crossover_rate


def generate_trials(
    target: FloatArray,
    strategy: MutationStrategy,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    rng: np.random.Generator,
    num_per_strategy: int = 1,
) -> FloatArray:
    """Apply one mutation strategy and the original binomial crossover."""

    trials: list[FloatArray] = []
    for _ in range(num_per_strategy):
        factor, crossover_rate = getF_CR(rng)
        mutant = strategy(factor)
        crossover_mask = rng.random(len(target)) < crossover_rate
        trial = np.where(crossover_mask, mutant, target)
        trials.append(np.clip(trial, lower_bounds, upper_bounds))
    return np.asarray(trials, dtype=float)


def generate_trial_candidates_group(
    population: FloatArray,
    target_index: int,
    objective: FloatArray,
    penalty: FloatArray,
    fitness: FloatArray,
    violations: FloatArray,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    rng: np.random.Generator,
    *,
    num_trials: int = 6,
    p_pbest: float = 0.2,
    unified_order: np.ndarray | None = None,
) -> FloatArray:
    """Generate the same diversity/convergence strategy mixture as SADE_TED.

    With the default six trials, three of four diversity strategies and three
    of six convergence strategies are sampled without replacement for every
    target individual.
    """

    n_population = len(population)
    other_indices = np.delete(np.arange(n_population), target_index)
    if len(other_indices) < 4:
        raise ValueError("At least five population members are required for DE donors.")
    a, b, c, d = population[rng.choice(other_indices, size=4, replace=False)]
    target = population[target_index]

    if unified_order is None:
        finite_objective = np.where(np.isfinite(objective), objective, np.inf)
        fbest = population[int(np.argmin(finite_objective))]
        minimum_penalty = np.min(penalty)
        penalty_best_indices = np.flatnonzero(penalty == minimum_penalty)
        gbest = population[int(rng.choice(penalty_best_indices))]
        violation_count = np.sum(violations > 0.0, axis=1)
        pbest_order = np.lexsort((fitness, violation_count))
    else:
        pbest_order = np.asarray(unified_order, dtype=int).reshape(-1)
        if len(pbest_order) != n_population or set(pbest_order) != set(
            range(n_population)
        ):
            raise ValueError("unified_order must be a permutation of the population.")
        best = population[int(pbest_order[0])]
        fbest = best
        gbest = best
    pbest_count = max(1, int(p_pbest * n_population))
    pbest = population[int(rng.choice(pbest_order[:pbest_count]))]
    mean_vector = np.mean(population, axis=0)

    diversity_strategies: list[MutationStrategy] = [
        lambda f: a + f * (b - c),
        lambda f: a + f * (b - c) + f * (d - target),
        lambda f: a + f * (pbest - a) + f * (b - c),
        lambda f: target + f * (a - target) + f * (b - c),
    ]
    convergence_strategies: list[MutationStrategy] = [
        lambda f: pbest + f * (a - b),
        lambda f: target + f * (pbest - target) + f * (a - b),
        lambda f: target + f * (pbest - target) + f * (a - b) + f * (c - d),
        lambda f: target + f * (fbest - target) + f * (a - b),
        lambda f: target + f * (mean_vector - target) + f * (a - b),
        lambda f: a + f * (gbest - b) + f * (c - d),
    ]

    diversity_count = min(num_trials // 2, len(diversity_strategies))
    convergence_count = min(num_trials - diversity_count, len(convergence_strategies))
    strategy_indices = [
        (
            diversity_strategies,
            rng.choice(len(diversity_strategies), diversity_count, replace=False),
        ),
        (
            convergence_strategies,
            rng.choice(len(convergence_strategies), convergence_count, replace=False),
        ),
    ]

    trials: list[FloatArray] = []
    for strategies, indices in strategy_indices:
        for index in np.atleast_1d(indices):
            trials.append(
                generate_trials(
                    target,
                    strategies[int(index)],
                    lower_bounds,
                    upper_bounds,
                    rng,
                )
            )
    return np.vstack(trials)
