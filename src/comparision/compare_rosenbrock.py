"""Compare optimization algorithms on the **Rosenbrock** function (2-D).

Metrics: Convergence Speed · Solution Quality · Computational Time · Robustness
Generates a 2x2 comparison plot.

Usage
-----
    cd src
    python -m comparision.compare_rosenbrock [--cycle 200] [--runs 10] [--seed 42] [--save]
    python -m comparision.compare_rosenbrock --save --output-dir outputs/run_01
    python -m comparision.compare_rosenbrock --save --save-json

Output
------
    --output-dir sets the output root directory.
    Figures are saved to <root>/figures and JSON files to <root>/data.
    If omitted, root defaults to current working directory.
"""

from __future__ import annotations

import argparse

from AIP.problems.continuous.rosenbrock import Rosenbrock
from comparision.comparison_utils import (
    run_comparison,
    plot_comparison,
    print_summary_table,
    tune_all_algorithms,
    load_tuned_config,
    make_output_path,
    make_data_output_path,
    save_results_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algorithm comparison on Rosenbrock (2-D)"
    )
    parser.add_argument("--cycle", type=int, default=200, help="Iterations per run")
    parser.add_argument("--runs", type=int, default=10, help="Independent runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--save", action="store_true", help="Save figure instead of showing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root; saves figures to <root>/figures and JSON to <root>/data (default: .)",
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save raw comparison results as JSON"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Force re-tuning via grid search (overwrites saved config)",
    )
    parser.add_argument(
        "--tune-runs",
        type=int,
        default=5,
        help="Independent runs per config during tuning (default: 5)",
    )
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
            problem=problem,
            cycle=args.cycle,
            n_runs=args.tune_runs,
            seed=args.seed,
        )
    else:
        tuned_params = load_tuned_config("Rosenbrock")
        if tuned_params:
            print(
                f"\n>>> Loaded tuned config for Rosenbrock ({len(tuned_params)} algos)"
            )
        else:
            print(
                "\n>>> No saved config found, using defaults. "
                "Use --tune to run parameter tuning."
            )

    results = run_comparison(
        problem=problem,
        cycle=args.cycle,
        n_runs=args.runs,
        seed=args.seed,
        tuned_params=tuned_params,
    )

    print_summary_table(results)

    save_path = (
        make_output_path("compare_rosenbrock.png", args.output_dir)
        if args.save
        else None
    )

    plot_comparison(
        results, problem_name="Rosenbrock", n_dim=n_dim, save_path=save_path
    )

    if args.save_json:
        saved_json = save_results_json(
            results,
            make_data_output_path("compare_rosenbrock.json", args.output_dir),
            metadata={
                "problem": "Rosenbrock",
                "problem_type": "continuous",
                "n_dim": n_dim,
                "cycle": args.cycle,
                "runs": args.runs,
                "seed": args.seed,
            },
        )
        print(f"\nResults JSON saved to: {saved_json}")


if __name__ == "__main__":
    main()
