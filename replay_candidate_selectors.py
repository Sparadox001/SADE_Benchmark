"""Replay simple selectors on saved candidate pools."""

from __future__ import annotations

import argparse
from pathlib import Path

from sade_benchmark.diagnostics.selector_replay import (
    replay_candidate_selectors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay simple selectors on saved SADE candidate pools."
    )
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in replay_candidate_selectors(args.result_root, args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
