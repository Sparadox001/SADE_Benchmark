# CEC2017 objective-cap pilot, median T0 v1

This opt-in suite derives 12 instances from seven retained CEC2017 problems in
10D and/or 30D. Each instance minimizes the unchanged original objective and
keeps every original inequality. One inequality is appended:

`g_cap(x) = f(x) - T0 <= 0`.

The original `cec2017` suite is unchanged. The implementation evaluates the
original problem once per point batch and reuses the resulting objective for
`g_cap`; evaluating this additional constraint does not incur another FE.

Frozen full-precision thresholds and calibration metadata are in
`sade_benchmark/benchmarks/data/cec2017_objcap_median_t0.json`. For each
problem/dimension, DSI was run five times with seeds 1–5 and 1000 FE. Only
points feasible under the original inequalities were pooled; exactly matching
decision vectors were deduplicated; the median of their objective values is
the frozen T0. C22-10D had four successful calibration runs. Six instances
without any originally feasible calibration points were excluded.

The suite is intentionally excluded from the `--suites all` shortcut to keep
existing CEC experiments unchanged. Select it explicitly:

`python run_experiment.py --suites cec2017_objcap --problems all --dimensions 10 30 --algorithms dsi --runs 1 --seed 101 --max-evals 300`

The supplied pilot configuration runs SADE, SADE with optional dynamic
tightening, and DSI with seed 101 and 300 FE per instance:

`python run_experiment.py --config configs/objective_cap/median_t0_v1_pilot.json`

This is an exploratory pilot, not a statistical comparison. Calibration runs
are not reused as test runs. Pooled feasible points are adaptive DSI samples,
not uniform feasible-domain samples, and some runs repeatedly evaluated the
same vector (especially C20). The three algorithms must use the same frozen
T0 for each instance; do not recalibrate it from their outcomes.

The later `sade_targeted` variant leaves the benchmark's fixed T0 and official
feasibility unchanged, but replaces the objective-cap column of its internal
search violations with `max(f(x) - active_limit, 0)`. It starts at T0 and can
only tighten that one existing constraint; no extra search constraint is
appended. Its default target index is `-1`, the final constraint in this suite.
The earlier `sade_dynamic` remains separate and still appends a virtual
objective constraint; do not interpret it as the targeted variant.
