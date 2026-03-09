"""Compare optimization algorithms on the **Rosenbrock** function (2-D).

Metrics: Convergence Speed · Solution Quality · Computational Time · Robustness
Generates a 2×2 comparison plot.

Usage
-----
    cd src
    python -m comparision.compare_rosenbrock [--cycle 200] [--runs 10] [--seed 42] [--save]
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from problems.continuous.Rosenbrock import Rosenbrock
from comparision.comparison_utils import (
    run_comparison, plot_comparison, print_summary_table,
    tune_all_algorithms, load_tuned_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm comparison on Rosenbrock (2-D)")
    parser.add_argument("--cycle", type=int, default=200, help="Iterations per run")
    parser.add_argument("--runs", type=int, default=10, help="Independent runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save", action="store_true", help="Save figure instead of showing")
    parser.add_argument("--tune", action="store_true",
                        help="Force re-tuning via grid search (overwrites saved config)")
    parser.add_argument("--tune-runs", type=int, default=5,
                        help="Independent runs per config during tuning (default: 5)")
    args = parser.parse_args()

    n_dim = 2
    problem = Rosenbrock(n_dim=n_dim)
    print(f"Problem : Rosenbrock  |  Dimensions : {n_dim}")
    print(f"Bounds  : {problem._bounds[0]}")
    print(f"Cycle   : {args.cycle}  |  Runs : {args.runs}")

    tuned_params = None
    if args.tune:
        print("\n>>> Running parameter tuning for all algorithms...")
        tuned_params = tune_all_algorithms(
            problem=problem, cycle=args.cycle,
            n_runs=args.tune_runs, seed=args.seed,
        )
    else:
        tuned_params = load_tuned_config("Rosenbrock")
        if tuned_params:
            print(f"\n>>> Loaded tuned config for Rosenbrock ({len(tuned_params)} algos)")
        else:
            print("\n>>> No saved config found, using defaults. "
                  "Use --tune to run parameter tuning.")

    results = run_comparison(
        problem=problem,
        cycle=args.cycle,
        n_runs=args.runs,
        seed=args.seed,
        tuned_params=tuned_params,
    )

    print_summary_table(results)

    save_path = os.path.join(
        os.path.dirname(__file__), "figures", "compare_rosenbrock.png"
    ) if args.save else None

    plot_comparison(results, problem_name="Rosenbrock", n_dim=n_dim, save_path=save_path)


if __name__ == "__main__":
    main()
