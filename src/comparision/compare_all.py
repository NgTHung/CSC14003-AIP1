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

import matplotlib
import matplotlib.figure
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


def _plot_row(
    axes,
    results: list[RunResult],
    problem_name: str,
    algo_names: list[str],
    colors: list[str],
) -> None:
    """Draw one row (4 sub-plots) for a single problem."""

    grouped = _group_by_algo(results)

    # 1. Convergence Speed
    ax = axes[0]
    for idx, name in enumerate(algo_names):
        if name not in grouped:
            continue
        curves = [r.fitness_curve for r in grouped[name]]
        max_len = max(len(c) for c in curves)
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        mean_c = np.nanmean(arr, axis=0)
        std_c = np.nanstd(arr, axis=0)
        x = np.arange(1, max_len + 1)
        ax.plot(x, mean_c, label=name, color=colors[idx], linewidth=1.2)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        color=colors[idx], alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylabel(problem_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid(True, alpha=0.25, which="both", linestyle="--")
    ax.tick_params(labelsize=7)

    # 2. Solution Quality
    ax = axes[1]
    box_data = [
        [r.best_fitness for r in grouped.get(n, [])] or [np.nan]
        for n in algo_names
    ]
    bp = ax.boxplot(box_data, tick_labels=algo_names, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

    # 3. Computational Time
    ax = axes[2]
    means_t = [np.mean([r.time_ms for r in grouped.get(n, [])]) if n in grouped else 0
               for n in algo_names]
    stds_t = [np.std([r.time_ms for r in grouped.get(n, [])]) if n in grouped else 0
              for n in algo_names]
    ax.bar(algo_names, means_t, yerr=stds_t, capsize=3,
           color=colors, alpha=0.65, edgecolor="black", linewidth=0.4)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

    # 4. Robustness (std dev)
    ax = axes[3]
    stds_f = [np.std([r.best_fitness for r in grouped.get(n, [])]) if n in grouped else 0
              for n in algo_names]
    ax.bar(algo_names, stds_f, color=colors, alpha=0.65,
           edgecolor="black", linewidth=0.4)
    ax.tick_params(axis="x", rotation=60, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")


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

    # ── Build one figure per problem: 1 row × 4 columns each ──────────
    n_algos = len(all_algo_names)
    colors = (_COLORS * ((n_algos // len(_COLORS)) + 1))[:n_algos]

    col_titles = ["Convergence Speed", "Solution Quality",
                  "Computational Time (ms)", "Robustness (Std Dev)"]

    figures: list[tuple[str, matplotlib.figure.Figure]] = []

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
