"""Compare optimization algorithms on the **Graph Coloring** problem.

Metrics: Best Fitness · Computational Time
Generates a 1×2 comparison plot.

Algorithms compared:
  Classical : DFS, BFS, UCS, Greedy Best-First, A*
  Local     : HC (Hill Climbing)
  Natural   : SA, HS, ABC, CS, FA

The fitness metric is the **number of edge-conflicts** (lower is better;
0 = legal coloring).

Usage
-----
    python -m src.comparision.compare_graph_coloring [--size medium] [--cycle 500] [--seed 42] [--save]
    python -m src.comparision.compare_graph_coloring --no-classical --save
    python -m src.comparision.compare_graph_coloring --tune --tune-runs 3 --save
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from problems.discrete.graph_coloring import GraphColoring
from comparision.comparison_utils_discrete import (
    run_comparison, plot_comparison, plot_convergence, print_summary_table,
    tune_all_algorithms, load_tuned_config, _CLASSICAL_ALGOS,
)

_SIZE_FACTORIES = {
    "tiny":   GraphColoring.create_tiny,
    "small":  GraphColoring.create_small,
    "medium": GraphColoring.create_medium,
    "large":  GraphColoring.create_large,
}

CONFIG_NAME = "GraphColoring"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Algorithm comparison on Graph Coloring")
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

    print(f"Problem  : Graph Coloring ({args.size})")
    print(f"Vertices : {problem.n_vertices}  {problem.vertex_names}")
    print(f"Colors   : {problem.n_colors}")
    print(f"Edges    : {len(problem.edges)}")
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
            print(f"\n>>> Loaded tuned config for {config_name} "
                  f"({len(tuned_params)} algos)")
        else:
            print("\n>>> No saved config found, using defaults. "
                  "Use --tune to run parameter tuning.")

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

    # Format solution: show vertex -> color mapping
    def _fmt_gc(sol):
        # Classical algos return a path (list of tuples); take last state
        if isinstance(sol, list) and sol and isinstance(sol[0], tuple):
            sol = sol[-1]
        arr = np.asarray(sol).flatten().astype(int)
        parts = [f"{problem.vertex_names[i]}=C{c}" for i, c in enumerate(arr)]
        return ", ".join(parts)

    print_summary_table(results, fitness_label="Conflicts",
                        format_solution=_fmt_gc)

    save_path = os.path.join(
        os.path.dirname(__file__), "figures",
        f"compare_graph_coloring_{args.size}.png",
    ) if args.save else None

    plot_comparison(
        results,
        problem_name="Graph Coloring",
        problem_desc=(f"{args.size}, {problem.n_vertices} vertices, "
                      f"{len(problem.edges)} edges, {problem.n_colors} colors"),
        save_path=save_path,
        fitness_label="Conflicts",
    )

    conv_save_path = os.path.join(
        os.path.dirname(__file__), "figures",
        f"convergence_graph_coloring_{args.size}.png",
    ) if args.save else None

    plot_convergence(
        results,
        problem_name="Graph Coloring",
        problem_desc=(f"{args.size}, {problem.n_vertices} vertices, "
                      f"{len(problem.edges)} edges, {problem.n_colors} colors"),
        save_path=conv_save_path,
        fitness_label="Conflicts",
    )


if __name__ == "__main__":
    main()
