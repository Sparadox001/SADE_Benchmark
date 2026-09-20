"""Extract the CEC2010 constants from the local DSI MATLAB reference.

This development helper generated ``sade_benchmark/benchmarks/data/cec2010.npz``.
The optimizer itself never reads or depends on the DSI directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def case_block(source: str, number: int) -> str:
    match = re.search(
        rf"(?ms)^\s*case\s+{number}\s*$\n(.*?)(?=^\s*case\s+\d+\s*$|^\s*end\s*$)",
        source,
    )
    if match is None:
        raise ValueError(f"Could not find case {number}.")
    return match.group(1)


def matlab_array(block: str, variable: str) -> np.ndarray:
    match = re.search(rf"\b{re.escape(variable)}\s*=\s*\[(.*?)\];", block, re.S)
    if match is None:
        raise ValueError(f"Could not find {variable}.")
    rows = match.group(1).replace("...", "").split(";")
    values = [[float(item) for item in NUMBER.findall(row)] for row in rows]
    values = [row for row in values if row]
    widths = {len(row) for row in values}
    if len(widths) != 1:
        raise ValueError(f"Rows in {variable} have inconsistent lengths: {widths}.")
    result = np.asarray(values, dtype=float)
    return result.reshape(-1) if result.shape[0] == 1 else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}
    for problem in (1, 7, 8, 13, 14, 15):
        arrays[f"o{problem:02d}"] = matlab_array(case_block(source, problem), "O")
    rotated = case_block(source, 8)
    arrays["rotation_10"] = matlab_array(rotated, "M1")
    arrays["rotation_30"] = matlab_array(rotated, "M2")

    if arrays["rotation_10"].shape != (10, 10):
        raise ValueError(f"Unexpected 10-D rotation shape: {arrays['rotation_10'].shape}.")
    if arrays["rotation_30"].shape != (30, 30):
        raise ValueError(f"Unexpected 30-D rotation shape: {arrays['rotation_30'].shape}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
