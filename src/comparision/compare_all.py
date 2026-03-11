"""Compare optimization algorithms across **all** continuous problems (2-D).

Runs the full benchmark suite on Ackley, Griewank, Rastrigin, Rosenbrock,
and Sphere, then generates a combined multi-row figure and a per-problem
ranking summary.

Metrics per problem: Convergence Speed · Solution Quality · Computational Time · Robustness

Usage
-----
    cd src
    python -m comparision.compare_all [--cycle 200] [--runs 10] [--seed 42] [--save]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import matplotlib.pyplot as plt

from AIP.problems.continuous.ackley import Ackley
from AIP.problems.continuous.griewank import Griewank
from AIP.problems.continuous.rastrigin import Rastrigin
from AIP.problems.continuous.Rosenbrock import Rosenbrock
from AIP.problems.continuous.sphere import Sphere

from comparision.comparison_utils import (
    run_comparison,
    plot_comparison,
    print_summary_table,
    tune_all_algorithms,
    load_tuned_config,
    _group_by_algo,
    _COLORS,
    RunResult,
)


# ── All five benchmark problems ──────────────────────────────────────
PROBLEMS = {
    "Ackley":     lambda d: Ackley(n_dim=d),
    "Griewank":   lambda d: Griewank(n_dim=d),
    "Rastrigin":  lambda d: Rastrigin(n_dim=d),
    "Rosenbrock": lambda d: Rosenbrock(n_dim=d),
    "Sphere":     lambda d: Sphere(n_dim=d),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined comparison across all continuous problems")
    parser.add_argument("--dim", type=int, default=2, help="Problem dimensionality")
    parser.add_argument("--cycle", type=int, default=200, help="Iterations per run")
    parser.add_argument("--runs", type=int, default=10, help="Independent runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save", action="store_true", help="Save figure instead of showing")
    parser.add_argument("--tune", action="store_true",
                        help="Tune algorithm parameters per problem via grid search")
    parser.add_argument("--tune-runs", type=int, default=5,
                        help="Independent runs per config during tuning (default: 5)")
    args = parser.parse_args()

    n_dim = args.dim
    all_results: dict[str, list[RunResult]] = {}
    all_algo_names: list[str] = []

    for prob_name, factory in PROBLEMS.items():
        prob = factory(n_dim)
        print(f"\n{'='*60}")
        print(f"  PROBLEM: {prob_name}  (dim={n_dim})")
        print(f"{'='*60}")

        tuned_params = None
        if args.tune:
            print(f"\n>>> Tuning parameters for {prob_name}...")
            tuned_params = tune_all_algorithms(
                problem=prob, cycle=args.cycle,
                n_runs=args.tune_runs, seed=args.seed,
            )
        else:
            tuned_params = load_tuned_config(prob_name)
            if tuned_params:
                print(f">>> Loaded tuned config for {prob_name} ({len(tuned_params)} algos)")
            else:
                print(f">>> No saved config for {prob_name}, using defaults.")

        res = run_comparison(
            problem=prob,
            cycle=args.cycle,
            n_runs=args.runs,
            seed=args.seed,
            tuned_params=tuned_params,
        )
        all_results[prob_name] = res
        print_summary_table(res)

        if not all_algo_names:
            all_algo_names = list(dict.fromkeys(r.algo_name for r in res))

    # ── Plot 4 separate figures per problem ────────────────────────
    for prob_name, res in all_results.items():
        save_path = None
        if args.save:
            fig_dir = os.path.join(os.path.dirname(__file__), "figures")
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(
                fig_dir, f"compare_{prob_name.lower()}_dim_{n_dim}.png"
            )
        plot_comparison(
            results=res,
            problem_name=prob_name,
            n_dim=n_dim,
            save_path=save_path,
        )

    # ── Final ranking table across all problems ──────────────────────
    print(f"\n{'='*80}")
    print("  OVERALL RANKING  (lower mean fitness = better)")
    print(f"{'='*80}")

    ranking: dict[str, list[float]] = {n: [] for n in all_algo_names}
    for prob_name, res in all_results.items():
        grouped = _group_by_algo(res)
        for name in all_algo_names:
            if name in grouped:
                mean_f = float(np.mean([r.best_fitness for r in grouped[name]]))
                ranking[name].append(mean_f)
            else:
                ranking[name].append(np.inf)

    prob_names = list(PROBLEMS.keys())
    header = f"  {'Algo':<8s}"
    for pn in prob_names:
        header += f" | {pn:>12s}"
    header += f" | {'Avg Rank':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Rank per problem (1 = best)
    rank_matrix = np.zeros((len(all_algo_names), len(prob_names)))
    for j in range(len(prob_names)):
        vals = [ranking[n][j] for n in all_algo_names]
        order = np.argsort(vals)
        for rank, idx in enumerate(order):
            rank_matrix[idx, j] = rank + 1

    for i, name in enumerate(all_algo_names):
        row = f"  {name:<8s}"
        for j in range(len(prob_names)):
            row += f" | {ranking[name][j]:>12.4e}"
        avg_rank = np.mean(rank_matrix[i])
        row += f" | {avg_rank:>8.2f}"
        print(row)

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
