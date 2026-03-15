"""Compare optimization algorithms on the **0/1 Knapsack** problem.

Metrics: Best Fitness · Computational Time
Generates a 1x2 comparison plot.

Algorithms compared:
  Classical : DFS, BFS, UCS, Greedy Best-First, A*
  Local     : HC (Hill Climbing)
  Natural   : SA, HS, GSA, ABC, CS, FA

Note: Knapsack uses *minimization* of ``-total_value + penalty``, so
lower fitness is better (higher profit).

Usage
-----
    python -m src.comparision.compare_knapsack [--size medium] [--cycle 500] [--runs 5] [--seed 42] [--save]
    python -m src.comparision.compare_knapsack --no-classical --save
    python -m src.comparision.compare_knapsack --tune --tune-runs 3 --save
    python -m src.comparision.compare_knapsack --save --save-json --output-dir outputs/run_01

Output
------
    --output-dir sets the output root directory.
    Figures are saved to <root>/figures and JSON files to <root>/data.
    If omitted, root defaults to current working directory.
"""

from __future__ import annotations

import argparse

import numpy as np
from AIP.problems.discrete.knapsack import Knapsack
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
    "tiny": Knapsack.create_tiny,
    "small": Knapsack.create_small,
    "medium": Knapsack.create_medium,
    "large": Knapsack.create_large,
}

CONFIG_NAME = "Knapsack"


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm comparison on 0/1 Knapsack")
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
        "(recommended for large instances)",
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

    print(f"Problem  : 0/1 Knapsack ({args.size})")
    print(f"Items    : {problem.n_items}")
    print(f"Capacity : {problem.capacity}")
    print(f"Weights  : {problem.weights.tolist()}")
    print(f"Values   : {problem.values.tolist()}")
    print(f"Cycle    : {args.cycle}")

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

    skip_algos = _CLASSICAL_ALGOS if args.no_classical else set()
    # BFS exhausts memory on medium/large instances
    if args.size in ("medium", "large") and "BFS" not in skip_algos:
        skip_algos = skip_algos | {"BFS"}
        print(">>> Skipping BFS (too memory-intensive for this size)")
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

    # Format solution: show selected item indices
    def _fmt_knapsack(sol):
        # Classical algos return a path (list of tuples); take last state
        if isinstance(sol, list) and sol and isinstance(sol[0], tuple):
            sol = sol[-1]
        arr = np.asarray(sol).flatten()
        selected = [str(i) for i, v in enumerate(arr) if v >= 0.5]
        return "Items: [" + ", ".join(selected) + "]"

    print_summary_table(
        results,
        negate_fitness=True,
        fitness_label="Profit",
        format_solution=_fmt_knapsack,
    )

    save_path = (
        make_output_path(f"compare_knapsack_{args.size}.png", args.output_dir)
        if args.save
        else None
    )

    plot_comparison(
        results,
        problem_name="0/1 Knapsack",
        problem_desc=f"{args.size}, {problem.n_items} items, cap={problem.capacity:.0f}",
        save_path=save_path,
        negate_fitness=True,
        fitness_label="Profit",
    )

    conv_save_path = (
        make_output_path(f"convergence_knapsack_{args.size}.png", args.output_dir)
        if args.save
        else None
    )

    plot_convergence(
        results,
        problem_name="0/1 Knapsack",
        problem_desc=f"{args.size}, {problem.n_items} items, cap={problem.capacity:.0f}",
        save_path=conv_save_path,
        negate_fitness=True,
        fitness_label="Profit",
    )

    if args.save_json:
        saved_json = save_results_json(
            results,
            make_data_output_path(
                f"compare_knapsack_{args.size}.json", args.output_dir
            ),
            metadata={
                "problem": "Knapsack",
                "problem_type": "discrete",
                "size": args.size,
                "n_items": problem.n_items,
                "capacity": float(problem.capacity),
                "cycle": args.cycle,
                "seed": args.seed,
            },
        )
        print(f"\nResults JSON saved to: {saved_json}")


if __name__ == "__main__":
    main()
