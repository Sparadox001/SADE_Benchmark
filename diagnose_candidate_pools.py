"""CLI for offline true-evaluation diagnosis of saved candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path

from sade_benchmark.diagnostics import analyze_candidate_pool_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved candidate pools with the true benchmark."
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = analyze_candidate_pool_results(args.result_root, args.output_dir)
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
