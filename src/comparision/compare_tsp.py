"""Compare optimization algorithms on the **Travelling Salesman Problem** (TSP).

Metrics: Best Fitness · Computational Time
Generates a 1x2 comparison plot.

Algorithms compared:
  Classical : DFS, BFS, UCS, Greedy Best-First, A*
  Local     : HC (Hill Climbing)
  Natural   : SA, HS, GSA, ABC, CS, FA, AS, ACS, MMAS

Usage
-----
    python -m src.comparision.compare_tsp [--size medium] [--cycle 500] [--runs 5] [--seed 42] [--save]
    python -m src.comparision.compare_tsp --no-classical --save
    python -m src.comparision.compare_tsp --tune --tune-runs 3 --save
    python -m src.comparision.compare_tsp --size large --timeout 30
    python -m src.comparision.compare_tsp --save --save-json --output-dir outputs/run_01

Output
------
    --output-dir sets the output root directory.
    Figures are saved to <root>/figures and JSON files to <root>/data.
    If omitted, root defaults to current working directory.
"""

from __future__ import annotations

import argparse

import numpy as np

from AIP.problems.discrete.tsp import TSP
from comparision.comparison_utils_discrete import (
    run_comparison,
    plot_comparison,
    plot_convergence,
    print_summary_table,
    tune_all_algorithms,
    load_tuned_config,
    _CLASSICAL_ALGOS,
    make_output_path,
    make_data_output_path,
    save_results_json,
)

_SIZE_FACTORIES = {
    "tiny": TSP.create_tiny,
    "small": TSP.create_small,
    "medium": TSP.create_medium,
    "large": TSP.create_large,
}

CONFIG_NAME = "TSP"


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm comparison on TSP")
    parser.add_argument(
        "--size",
        type=str,
        default="medium",
        choices=list(_SIZE_FACTORIES.keys()),
        help="Problem instance size (default: medium)",
    )
    parser.add_argument(
        "--cycle", type=int, default=500, help="Iterations per run (default: 500)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Independent runs per stochastic algorithm (default: 5)",
    )
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
        "--tune", action="store_true", help="Force re-tuning via grid search"
    )
    parser.add_argument(
        "--tune-runs",
        type=int,
        default=3,
        help="Independent runs per config during tuning",
    )
    parser.add_argument(
        "--no-classical",
        action="store_true",
        help="Skip classical graph-search algorithms "
        "(recommended for medium/large instances)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Max seconds per algorithm run (default: 60)",
    )
    args = parser.parse_args()

    problem = _SIZE_FACTORIES[args.size]()
    config_name = f"{CONFIG_NAME}_{args.size}"

    print(f"Problem : TSP ({args.size})")
    print(f"Cities  : {problem.n_cities}  {problem.city_names}")
    print(f"Cycle   : {args.cycle}")

    tuned_params = None
    if args.tune:
        print("\n>>> Running parameter tuning for all algorithms...")
        tuned_params = tune_all_algorithms(
            problem=problem,
            problem_config_name=config_name,
            cycle=args.cycle,
            n_runs=args.tune_runs,
            seed=args.seed,
        )
    else:
        tuned_params = load_tuned_config(config_name)
        if tuned_params:
            print(
                f"\n>>> Loaded tuned config for {config_name} "
                f"({len(tuned_params)} algos)"
            )
        else:
            print(
                "\n>>> No saved config found, using defaults. "
                "Use --tune to run parameter tuning."
            )

    # Filter out classical algos if requested
    skip_algos = _CLASSICAL_ALGOS if args.no_classical else set()
    if skip_algos:
        print(f">>> Skipping: {', '.join(sorted(skip_algos))}")

    results = run_comparison(
        problem=problem,
        cycle=args.cycle,
        n_runs=args.runs,
        seed=args.seed,
        tuned_params=tuned_params,
        skip_algos=skip_algos,
        timeout=args.timeout,
    )

    # Format solution: show city-name path
    def _fmt_tsp(sol):
        # Classical algos return a path of (city, frozenset) state tuples
        if isinstance(sol, list) and sol and isinstance(sol[0], tuple):
            cities = [s[0] for s in sol]
            names = [problem.city_names[i] for i in cities]
            return " -> ".join(names)
        arr = np.asarray(sol).flatten().astype(int)
        names = [problem.city_names[i] for i in arr]
        return " -> ".join(names)

    print_summary_table(results, fitness_label="Distance", format_solution=_fmt_tsp)

    save_path = (
        make_output_path(f"compare_tsp_{args.size}.png", args.output_dir)
        if args.save
        else None
    )

    plot_comparison(
        results,
        problem_name="TSP",
        problem_desc=f"{args.size}, {problem.n_cities} cities",
        save_path=save_path,
        fitness_label="Distance",
    )

    conv_save_path = (
        make_output_path(f"convergence_tsp_{args.size}.png", args.output_dir)
        if args.save
        else None
    )

    plot_convergence(
        results,
        problem_name="TSP",
        problem_desc=f"{args.size}, {problem.n_cities} cities",
        save_path=conv_save_path,
        fitness_label="Distance",
    )

    if args.save_json:
        saved_json = save_results_json(
            results,
            make_data_output_path(f"compare_tsp_{args.size}.json", args.output_dir),
            metadata={
                "problem": "TSP",
                "problem_type": "discrete",
                "size": args.size,
                "n_cities": problem.n_cities,
                "cycle": args.cycle,
                "seed": args.seed,
            },
        )
        print(f"\nResults JSON saved to: {saved_json}")


if __name__ == "__main__":
    main()
