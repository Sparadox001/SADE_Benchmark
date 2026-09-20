# CEC2017 objective cap: first-feasible-mean T0 v1

This opt-in suite retains the same 12 CEC2017-derived instances as the
median-T0 suite. It keeps the original objective and inequalities, and adds
the single fixed inequality `f(x) - T0 <= 0`. The original CEC2017 suite and
the median-T0 suite are unchanged.

For each instance, T0 is the arithmetic mean of the objective values at the
first originally feasible evaluation in five independent DSI calibration
runs (seeds 1–5, 1000 FE each). A run with no originally feasible evaluation
does not contribute to the mean. C22-10D therefore uses four values; the
other 11 instances use five. Full-precision frozen thresholds and provenance
are in `sade_benchmark/benchmarks/data/cec2017_objcap_first_mean_t0.json`.

The supplied pilot uses new seed 101, 300 FE, and compares `sade`,
`sade_targeted`, and `dsi`:

`python run_experiment.py --config configs/objective_cap/first_feasible_mean_t0_v1_pilot.json`

`sade_targeted` internally tightens the existing objective-cap constraint
without appending another search constraint. Official feasibility always uses
the fixed problem T0, so all algorithms are evaluated on the same problem.
The first-feasible-mean cap can be very loose; the pilot is needed to measure
whether this constraint is informative at 300 FE. One seed cannot support a
statistical algorithm comparison.

## Seed-101 pilot result (300 FE)

Output: `results/objective_cap/first_feasible_mean_t0_v1_pilot_20260919_seed101`.
All 36 runs completed with 300 evaluations. Feasible-instance counts were
12/12 for `sade`, 12/12 for `sade_targeted`, and 10/12 for `dsi` (DSI did not
find a feasible point on C13-10D or C22-10D). Relative to `sade`, the
targeted variant obtained a lower feasible objective on 9 instances and tied
on 3; it tightened the existing cap on 10 instances. Relative to DSI, the
targeted variant won 5 and lost 7 instances under feasible-first comparison.
In particular, DSI remained much better on C05-10D (0.729 vs 401.2).

The starting cap is loose on several instances, although the original
inequalities still matter: the shared initial 30 evaluations contained zero
new-problem-feasible points on C02-10D/30D, C05-10D/30D, C13-10D, and
C22-10D. Every recorded cap constraint was verified against `f - T0`.
For SADE, the cap delayed first feasibility relative to original-constraint
feasibility on 4/12 instances, so it was not entirely inactive.
These are exploratory, single-seed results rather than statistical evidence.
