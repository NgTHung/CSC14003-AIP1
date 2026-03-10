"""Compare optimization algorithms on the **Travelling Salesman Problem** (TSP).

Metrics: Best Fitness · Computational Time
Generates a 1×2 comparison plot.

Algorithms compared:
  Classical : DFS, BFS, UCS, Greedy Best-First, A*
  Local     : HC (Hill Climbing)
  Natural   : SA, HS, ABC, CS, FA, AS, ACS, MMAS

Usage
-----
    python -m src.comparision.compare_tsp [--size medium] [--cycle 500] [--seed 42] [--save]
    python -m src.comparision.compare_tsp --no-classical --save
    python -m src.comparision.compare_tsp --tune --tune-runs 3 --save
    python -m src.comparision.compare_tsp --size large --timeout 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from problems.discrete.tsp import TSP
from comparision.comparison_utils_discrete import (
    run_comparison, plot_comparison, plot_convergence, print_summary_table,
    tune_all_algorithms, load_tuned_config, _CLASSICAL_ALGOS,
)

_SIZE_FACTORIES = {
    "tiny":   TSP.create_tiny,
    "small":  TSP.create_small,
    "medium": TSP.create_medium,
    "large":  TSP.create_large,
}

CONFIG_NAME = "TSP"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algorithm comparison on TSP")
    parser.add_argument("--size", type=str, default="medium",
                        choices=list(_SIZE_FACTORIES.keys()),
                        help="Problem instance size (default: medium)")
    parser.add_argument("--cycle", type=int, default=500,
                        help="Iterations per run (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--save", action="store_true",
                        help="Save figure instead of showing")
    parser.add_argument("--tune", action="store_true",
                        help="Force re-tuning via grid search")
    parser.add_argument("--tune-runs", type=int, default=3,
                        help="Independent runs per config during tuning")
    parser.add_argument("--no-classical", action="store_true",
                        help="Skip classical graph-search algorithms "
                             "(recommended for medium/large instances)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Max seconds per algorithm run (default: 60)")
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
            print(f"\n>>> Loaded tuned config for {config_name} "
                  f"({len(tuned_params)} algos)")
        else:
            print("\n>>> No saved config found, using defaults. "
                  "Use --tune to run parameter tuning.")

    # Filter out classical algos if requested
    skip_algos = _CLASSICAL_ALGOS if args.no_classical else set()
    if skip_algos:
        print(">>> Skipping classical graph-search algorithms")

    results = run_comparison(
        problem=problem,
        cycle=args.cycle,
        seed=args.seed,
        tuned_params=tuned_params,
        skip_algos=skip_algos,
        timeout=args.timeout,
    )

    # Format solution: show city-name path
    def _fmt_tsp(sol):
        # Classical algos return a path (list of tuples); take last state
        if isinstance(sol, list) and sol and isinstance(sol[0], tuple):
            sol = sol[-1]
        arr = np.asarray(sol).flatten().astype(int)
        names = [problem.city_names[i] for i in arr]
        return " -> ".join(names)

    print_summary_table(results, fitness_label="Distance",
                        format_solution=_fmt_tsp)

    save_path = os.path.join(
        os.path.dirname(__file__), "figures", f"compare_tsp_{args.size}.png"
    ) if args.save else None

    plot_comparison(
        results,
        problem_name="TSP",
        problem_desc=f"{args.size}, {problem.n_cities} cities",
        save_path=save_path,
        fitness_label="Distance",
    )

    conv_save_path = os.path.join(
        os.path.dirname(__file__), "figures",
        f"convergence_tsp_{args.size}.png",
    ) if args.save else None

    plot_convergence(
        results,
        problem_name="TSP",
        problem_desc=f"{args.size}, {problem.n_cities} cities",
        save_path=conv_save_path,
        fitness_label="Distance",
    )


if __name__ == "__main__":
    main()
