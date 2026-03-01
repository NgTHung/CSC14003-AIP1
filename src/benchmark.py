"""Benchmarking script for Human Behavior-Based Algorithms (CA, SFO, TLBO).

Runs each algorithm multiple independent times on a chosen continuous
benchmark problem, records execution time and best fitness, then exports
raw results to CSV and prints a summary table.

Usage
-----
    python benchmark.py [--seed SEED] [--dim DIM] [--cycle CYCLE]
                        [--problem PROBLEM] [--runs RUNS]
"""

import argparse
import csv
import random
import time

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from problems import Sphere, Rastrigin, Ackley, Cigar, Ridge
from algorithm.natural.human.ca import CA, CAConfig
from algorithm.natural.human.sfo import SFO, SFOConfig
from algorithm.natural.human.tlbo import TLBO, TLBOConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int | None) -> None:
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
        print(f"Random seed set to: {seed}")


# ---------------------------------------------------------------------------
# Algorithm factory — maps a name to (ConfigClass, AlgoClass, default_kwargs)
# ---------------------------------------------------------------------------

ALGO_REGISTRY: dict[str, tuple[type, type, dict]] = {
    "CA": (
        CAConfig,
        CA,
        {
            "pop_size": 50,
            "accepted_ratio": 0.2,
            "exploit_ratio": 0.8,
            "explore_sigma": 0.1,
        },
    ),
    "SFO": (
        SFOConfig,
        SFO,
        {
            "pop_size": 50,
            "w": 0.9,
            "w_decay": 0.99,
            "c_attract": 1.5,
            "c_social": 1.5,
        },
    ),
    "TLBO": (
        TLBOConfig,
        TLBO,
        {
            "pop_size": 50,
        },
    ),
}


def run_benchmark(
    problem,
    problem_name: str,
    n_dim: int,
    cycle: int,
    n_runs: int,
) -> list[dict]:
    """Run all registered algorithms and collect per-run statistics.

    Parameters
    ----------
    problem : ContinuousProblem
        The benchmark problem instance.
    problem_name : str
        Human-readable name of the problem (for CSV).
    n_dim : int
        Dimensionality of the search space.
    cycle : int
        Number of iterations per algorithm run.
    n_runs : int
        Number of independent repetitions per algorithm.

    Returns
    -------
    list[dict]
        One entry per (algorithm, run) with keys:
        ``Algorithm``, ``Problem``, ``Dimensions``, ``Run_ID``,
        ``Best_Fitness``, ``Execution_Time_ms``, and ``history``
        (convergence curve kept in memory for plotting).
    """
    rows: list[dict] = []

    for algo_name, (ConfigCls, AlgoCls, default_kwargs) in ALGO_REGISTRY.items():
        print(f"\n── {algo_name} ({n_runs} runs) ──")

        for run_id in range(1, n_runs + 1):
            # Build config with the default hyperparameters
            cfg = ConfigCls(
                iterations=cycle,
                minimization=True,
                **default_kwargs,
            )
            model = AlgoCls(configuration=cfg, problem=problem)

            # ── Time only the core optimisation process ──
            t_start = time.perf_counter()
            model.run()
            t_end = time.perf_counter()

            elapsed_ms = (t_end - t_start) * 1000.0  # convert to ms

            rows.append(
                {
                    "Algorithm": algo_name,
                    "Problem": problem_name,
                    "Dimensions": n_dim,
                    "Run_ID": run_id,
                    "Best_Fitness": float(model.best_fitness),
                    "Execution_Time_ms": elapsed_ms,
                    "history": list(model.history),  # kept in memory only
                }
            )

            print(
                f"  Run {run_id:>{len(str(n_runs))}}/{n_runs}  |  "
                f"Fitness: {model.best_fitness:.10f}  |  "
                f"Time: {elapsed_ms:>10.3f} ms"
            )

    return rows


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "Algorithm",
    "Problem",
    "Dimensions",
    "Run_ID",
    "Best_Fitness",
    "Execution_Time_ms",
]


def export_csv(rows: list[dict], filepath: str = "benchmark_results_time.csv") -> None:
    """Write benchmark results to a CSV file.

    Parameters
    ----------
    rows : list[dict]
        Output of :func:`run_benchmark`.
    filepath : str
        Destination CSV filename.  Default ``benchmark_results_time.csv``.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults exported to: {filepath}")


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    """Print a Markdown-style summary table to the terminal.

    Columns: Algorithm | Mean Fitness | Std Dev | Mean Time (ms)
    """
    # Aggregate per algorithm
    algo_names: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r["Algorithm"] not in seen:
            algo_names.append(r["Algorithm"])
            seen.add(r["Algorithm"])

    print(f"\n{'=' * 72}")
    print(f"  {'Algorithm':<12s} | {'Mean Fitness':>18s} | {'Std Dev':>14s} | {'Mean Time (ms)':>16s}")
    print(f"  {'-' * 12}-+-{'-' * 18}-+-{'-' * 14}-+-{'-' * 16}")

    for name in algo_names:
        fits  = [r["Best_Fitness"]      for r in rows if r["Algorithm"] == name]
        times = [r["Execution_Time_ms"] for r in rows if r["Algorithm"] == name]
        mean_fit  = float(np.mean(fits))
        std_fit   = float(np.std(fits))
        mean_time = float(np.mean(times))
        print(
            f"  {name:<12s} | {mean_fit:>18.10f} | {std_fit:>14.10f} | {mean_time:>16.3f}"
        )

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Plotting (convergence curves & box plot)
# ---------------------------------------------------------------------------

def plot_results(rows: list[dict], n_dim: int, problem_name: str) -> None:
    """Display convergence curves and a box plot of final fitness values.

    Parameters
    ----------
    rows : list[dict]
        Output of :func:`run_benchmark` (must still contain ``history``).
    n_dim : int
        Problem dimensionality (for titles).
    problem_name : str
        Name of the benchmark problem (for titles).
    """
    if not HAS_PLT:
        print("\nmatplotlib not installed — skipping plots.")
        return

    # Collect per-algorithm histories and final fitness values
    algo_names: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r["Algorithm"] not in seen:
            algo_names.append(r["Algorithm"])
            seen.add(r["Algorithm"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Subplot 1: Convergence curves (mean ± std across runs) ──
    ax1 = axes[0]
    for name in algo_names:
        histories = [r["history"] for r in rows if r["Algorithm"] == name]
        # Pad to the same length (in case some runs differ negligibly)
        max_len = max(len(h) for h in histories)
        arr = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            arr[i, : len(h)] = h
        mean_curve = np.nanmean(arr, axis=0)
        std_curve  = np.nanstd(arr, axis=0)
        x = np.arange(max_len)
        ax1.plot(x, mean_curve, label=name, linewidth=1.5)
        ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

    ax1.set_title(f"Convergence on {problem_name} (dim={n_dim})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Fitness")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both", linestyle="--")
    ax1.legend(fontsize=9)

    # ── Subplot 2: Box plot of final best fitness ──
    ax2 = axes[1]
    box_data = []
    box_labels = []
    for name in algo_names:
        fits = [r["Best_Fitness"] for r in rows if r["Algorithm"] == name]
        box_data.append(fits)
        box_labels.append(name)

    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_title(f"Best Fitness Distribution ({problem_name}, dim={n_dim})")
    ax2.set_ylabel("Best Fitness")
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CA, SFO, TLBO on continuous problems"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dim", type=int, default=10,
        help="Problem dimensionality (default: 10)",
    )
    parser.add_argument(
        "--cycle", type=int, default=500,
        help="Number of iterations per run (default: 500)",
    )
    parser.add_argument(
        "--problem", type=str, default="ackley",
        choices=["sphere", "rastrigin", "ackley", "cigar", "ridge"],
        help="Benchmark problem (default: ackley)",
    )
    parser.add_argument(
        "--runs", type=int, default=30,
        help="Number of independent runs per algorithm (default: 30)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Select problem
    problems = {
        "sphere": Sphere,
        "rastrigin": Rastrigin,
        "ackley": Ackley,
        "cigar": Cigar,
        "ridge": Ridge,
    }
    problem = problems[args.problem](n_dim=args.dim)
    print(f"Problem: {problem._name}, Dimensions: {problem._n_dim}")
    print(f"Bounds : {problem._bounds[0]}")

    # ── Execute benchmark ────────────────────────────────────────────
    rows = run_benchmark(
        problem=problem,
        problem_name=args.problem,
        n_dim=args.dim,
        cycle=args.cycle,
        n_runs=args.runs,
    )

    # ── Export CSV ───────────────────────────────────────────────────
    export_csv(rows)

    # ── Terminal summary ─────────────────────────────────────────────
    print_summary(rows)

    # ── Plots ────────────────────────────────────────────────────────
    plot_results(rows, n_dim=args.dim, problem_name=args.problem)


if __name__ == "__main__":
    main()
