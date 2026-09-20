"""Numerically compare every retained Python evaluator with local DSI MATLAB code."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sade_benchmark.benchmarks import BENCHMARKS, make_problem  # noqa: E402


def matlab_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dsi", type=Path)
    parser.add_argument("--matlab", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(20260913)
    inputs: dict[str, np.ndarray] = {}
    expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    commands: list[str] = []
    output_names: list[str] = []

    def add_case(suite: str, name: str, dimension: int | None, index: int) -> None:
        problem = make_problem(suite, name, dimension)
        key = f"{suite}_{name}_d{problem.dimension}"
        unit = 0.1 + 0.8 * rng.random((3, problem.dimension))
        points = problem.lower_bounds + unit * (problem.upper_bounds - problem.lower_bounds)
        inputs[f"x_{key}"] = points
        expected[key] = problem.evaluate(points)
        if suite == "cec2006":
            commands.append(
                f"[~,~,aaa,~]=problemsetting({index}); "
                f"[f_{key},g_{key},~]=fitness_2006(x_{key},{index},aaa);"
            )
        elif suite == "cec2010":
            commands.append(
                f"[f_{key},g_{key},~]=fitness_2010(x_{key},{index},[]);"
            )
        else:
            commands.append(
                f"clear fitness_2017; "
                f"[f_{key},g_{key},~]=fitness_2017(x_{key},{index},0);"
            )
        output_names.extend((f"f_{key}", f"g_{key}"))

    for name in BENCHMARKS["cec2006"]["problems"]:
        add_case("cec2006", name, None, int(name[1:]))
    for dimension in BENCHMARKS["cec2010"]["dimensions"]:
        for name in BENCHMARKS["cec2010"]["problems"]:
            add_case("cec2010", name, int(dimension), int(name[1:]))
    for dimension in BENCHMARKS["cec2017"]["dimensions"]:
        for name in BENCHMARKS["cec2017"]["problems"]:
            add_case("cec2017", name, int(dimension), int(name[1:]))

    with tempfile.TemporaryDirectory(prefix="sade_dsi_check_") as temporary:
        temporary_path = Path(temporary)
        input_file = temporary_path / "inputs.mat"
        output_file = temporary_path / "outputs.mat"
        error_file = temporary_path / "matlab_error.txt"
        script_file = temporary_path / "validate.m"
        # The local DSI folder calls this shared file by its official name but
        # stores the byte-identical data as Function2.mat.
        shutil.copyfile(args.dsi / "Function2.mat", temporary_path / "ShiftAndRotation.mat")
        savemat(input_file, inputs)
        save_names = ", ".join(f"'{name}'" for name in output_names)
        script = "\n".join(
            (
                f"cd('{matlab_path(args.dsi)}');",
                f"addpath('{matlab_path(temporary_path)}');",
                f"load('{matlab_path(input_file)}');",
                *commands,
                f"save('{matlab_path(output_file)}', {save_names}, '-v7');",
            )
        )
        script_file.write_text(script, encoding="utf-8")
        expression = (
            f"try, run('{matlab_path(script_file)}'); "
            f"catch err, fid=fopen('{matlab_path(error_file)}','w'); "
            "fprintf(fid,'%s',getReport(err)); fclose(fid); exit(1); end; exit(0);"
        )
        completed = subprocess.run(
            [
                str(args.matlab),
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-wait",
                "-r",
                expression,
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=300,
        )
        if completed.returncode != 0 or not output_file.exists():
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            if error_file.exists():
                print(error_file.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            raise SystemExit(completed.returncode or 1)
        actual = loadmat(output_file)

    failures: list[str] = []
    for key, (expected_f, expected_g) in expected.items():
        actual_f = np.asarray(actual[f"f_{key}"], dtype=float).reshape(-1)
        actual_g = np.asarray(actual[f"g_{key}"], dtype=float)
        if actual_g.ndim == 1:
            actual_g = actual_g.reshape(-1, 1)
        f_ok = np.allclose(expected_f, actual_f, rtol=1e-10, atol=1e-8, equal_nan=True)
        g_ok = np.allclose(expected_g, actual_g, rtol=1e-10, atol=1e-8, equal_nan=True)
        if not (f_ok and g_ok):
            f_error = float(np.nanmax(np.abs(expected_f - actual_f)))
            g_error = float(np.nanmax(np.abs(expected_g - actual_g)))
            failures.append(f"{key}: max |df|={f_error:.3e}, max |dg|={g_error:.3e}")

    if failures:
        print("Validation failures:")
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"Validated {len(expected)} problem/dimension combinations against DSI MATLAB.")


if __name__ == "__main__":
    main()
