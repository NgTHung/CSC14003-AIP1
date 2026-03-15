"""Profiler script to identify performance bottlenecks in algorithm implementations.

Continuous mode (default):
    python profiler.py                      # Profile all algorithms (Sphere 10-D, 100 iters)
    python profiler.py --algo PSO ABC       # Profile specific algorithms
    python profiler.py --dim 30 --cycle 200 # Custom dimensionality and iterations

Discrete mode:
    python profiler.py --discrete tsp                # TSP with 15 random cities
    python profiler.py --discrete knapsack            # Knapsack with 20 items
    python profiler.py --discrete graph_coloring      # Graph coloring (10 vertices, 4 colors)
    python profiler.py --discrete tsp --n 25          # TSP with 25 cities
    python profiler.py --discrete tsp --algo ABC CS   # Profile specific algorithms on TSP

Common options:
    python profiler.py --top 30             # Show top 30 functions per algorithm
    python profiler.py --save               # Save detailed profile reports to profiles/
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import io
import time
import random

import numpy as np

# ── Ensure ``src/`` is on the import path ────────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from AIP.problems.continuous.sphere import Sphere
from comparision.comparison_utils import (
    build_algo as build_algo_continuous,
    ALGO_REGISTRY as ALGO_REGISTRY_CONTINUOUS,
)
from comparision.comparison_utils_discrete import (
    build_algo as build_algo_discrete,
    ALGO_REGISTRY,
    ALGO_REGISTRY_COMMON as ALGO_REGISTRY_DISCRETE_COMMON,
    _NEEDS_INITIAL_STATE,
)
from AIP.problems.discrete.tsp import TSP
from AIP.problems.discrete.knapsack import Knapsack
from AIP.problems.discrete.graph_coloring import GraphColoring


def profile_algorithm(
    algo_name: str,
    problem,
    cycle: int,
    top_n: int,
    save_dir: str | None,
    *,
    build_fn,
    needs_initial_state: bool = False,
) -> dict:
    """Profile a single algorithm and return summary statistics.

    Parameters
    ----------
    algo_name : str
        Algorithm key.
    problem : Problem
        The benchmark problem instance.
    cycle : int
        Number of iterations per run.
    top_n : int
        Number of top functions to display.
    save_dir : str or None
        If given, save raw profile data to this directory.
    build_fn : callable
        Function ``(algo_name, params, problem, cycle) -> model``.
    needs_initial_state : bool
        If True, pass ``initial_state`` to ``model.run()``.

    Returns
    -------
    dict
        Summary with total time, top functions, and call counts.
    """
    np.random.seed(42)
    random.seed(42)
    model = build_fn(algo_name, {}, problem, cycle)

    profiler = cProfile.Profile()

    # Time the run with wall clock too
    t0 = time.perf_counter()
    profiler.enable()
    if needs_initial_state:
        initial = problem.sample(1).flatten()
        model.run(initial_state=initial)
    else:
        model.run()
    profiler.disable()
    wall_time = time.perf_counter() - t0

    # Capture stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # Save raw profile if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = algo_name.replace("/", "_").replace("+", "plus")
        profiler.dump_stats(os.path.join(save_dir, f"{safe_name}.prof"))

    # Print header
    print(f"\n{'=' * 75}")
    print(f"  {algo_name}  |  Wall time: {wall_time * 1000:.1f} ms  |  Cycle: {cycle}")
    print(f"{'=' * 75}")

    # Print top functions by cumulative time
    stats.print_stats(top_n)
    print(stream.getvalue())

    # Collect summary
    all_stats = stats.stats  # dict of (file, line, func) -> (cc, nc, tt, ct, callers)
    top_funcs = []
    sorted_items = sorted(all_stats.items(), key=lambda x: x[1][3], reverse=True)
    for (file, line, func), (cc, nc, tt, ct, _callers) in sorted_items[:top_n]:
        top_funcs.append({
            "function": f"{file}:{line}({func})",
            "calls": nc,
            "tottime_ms": tt * 1000,
            "cumtime_ms": ct * 1000,
        })

    return {
        "algo": algo_name,
        "wall_time_ms": wall_time * 1000,
        "total_calls": sum(v[1] for v in all_stats.values()),
        "top_functions": top_funcs,
    }


def print_summary_table(summaries: list[dict]) -> None:
    """Print a compact comparison table sorted by wall time."""
    print(f"\n{'=' * 75}")
    print(f"  SUMMARY — sorted by wall time (descending)")
    print(f"{'=' * 75}")
    print(f"  {'Algorithm':<16} {'Wall (ms)':>10} {'Total Calls':>14} {'Top Bottleneck'}")
    print(f"  {'-' * 16} {'-' * 10} {'-' * 14} {'-' * 30}")

    for s in sorted(summaries, key=lambda x: x["wall_time_ms"], reverse=True):
        bottleneck = s["top_functions"][0]["function"] if s["top_functions"] else "—"
        # Trim long function names
        if len(bottleneck) > 45:
            bottleneck = "..." + bottleneck[-42:]
        print(
            f"  {s['algo']:<16} {s['wall_time_ms']:>10.1f} "
            f"{s['total_calls']:>14,} {bottleneck}"
        )

    print()


# =====================================================================
# Discrete problem factories
# =====================================================================

def _make_tsp(n: int) -> TSP:
    """Create a random TSP instance with *n* cities."""
    return TSP.create_medium()
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(n, 2))
    dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    return TSP(dist)


def _make_knapsack(n: int) -> Knapsack:
    """Create a random Knapsack instance with *n* items."""
    rng = np.random.default_rng(0)
    weights = rng.uniform(1, 20, size=n)
    values = rng.uniform(5, 50, size=n)
    capacity = float(weights.sum() * 0.4)
    return Knapsack(weights, values, capacity)


def _make_graph_coloring(n: int, n_colors: int = 4) -> GraphColoring:
    """Create a random graph coloring instance with *n* vertices."""
    rng = np.random.default_rng(0)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.4:
                adj[i, j] = adj[j, i] = 1
    return GraphColoring(adj, n_colors)


def main():
    parser = argparse.ArgumentParser(
        description="Profile optimization algorithms to find bottlenecks.",
    )
    parser.add_argument(
        "--algo", nargs="*", default=None,
        help="Algorithm names to profile (default: all). E.g. --algo PSO ABC GA",
    )
    parser.add_argument(
        "--dim", type=int, default=10,
        help="Problem dimensionality for continuous mode (default: 10)",
    )
    parser.add_argument(
        "--cycle", type=int, default=100,
        help="Number of iterations (default: 100)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Top N functions to show per algorithm (default: 20)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save .prof files to profiles/ directory for later analysis",
    )
    parser.add_argument(
        "--discrete", type=str, default=None,
        choices=["tsp", "knapsack", "graph_coloring"],
        help="Run in discrete mode with the specified problem type",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Problem size for discrete mode: cities (TSP, default 15), "
             "items (Knapsack, default 20), vertices (GraphColoring, default 10)",
    )
    args = parser.parse_args()

    save_dir = os.path.join(os.path.dirname(__file__), "profiles") if args.save else None

    if args.discrete:
        # ── Discrete mode ────────────────────────────────────────────
        ptype = args.discrete.lower()
        if ptype == "tsp":
            n = args.n or 15
            problem = _make_tsp(n)
            registry = ALGO_REGISTRY
            label = f"TSP ({problem.n_cities} cities)"
        elif ptype == "knapsack":
            n = args.n or 20
            problem = _make_knapsack(n)
            registry = ALGO_REGISTRY_DISCRETE_COMMON
            label = f"Knapsack ({n} items)"
        elif ptype == "graph_coloring":
            n = args.n or 10
            problem = _make_graph_coloring(n)
            registry = ALGO_REGISTRY_DISCRETE_COMMON
            label = f"GraphColoring ({n} vertices, 4 colors)"
        else:
            raise ValueError(f"Unknown discrete problem: {ptype}")

        algo_names = args.algo if args.algo else list(registry.keys())

        print(f"\nProfiler [discrete] — {label}, {args.cycle} iterations")
        print(f"Algorithms: {', '.join(algo_names)}")
        if save_dir:
            print(f"Saving .prof files to: {os.path.abspath(save_dir)}")

        summaries = []
        for name in algo_names:
            if name not in registry:
                print(f"\n  [SKIP] Unknown algorithm: {name}")
                continue
            summary = profile_algorithm(
                name, problem, args.cycle, args.top, save_dir,
                build_fn=build_algo_discrete,
                needs_initial_state=(name in _NEEDS_INITIAL_STATE),
            )
            summaries.append(summary)
    else:
        # ── Continuous mode (original) ───────────────────────────────
        algo_names = args.algo if args.algo else list(ALGO_REGISTRY_CONTINUOUS.keys())
        problem = Sphere(n_dim=args.dim)

        print(f"\nProfiler [continuous] — Sphere ({args.dim}-D), {args.cycle} iterations")
        print(f"Algorithms: {', '.join(algo_names)}")
        if save_dir:
            print(f"Saving .prof files to: {os.path.abspath(save_dir)}")

        summaries = []
        for name in algo_names:
            if name not in ALGO_REGISTRY_CONTINUOUS:
                print(f"\n  [SKIP] Unknown algorithm: {name}")
                continue
            summary = profile_algorithm(
                name, problem, args.cycle, args.top, save_dir,
                build_fn=build_algo_continuous,
            )
            summaries.append(summary)

    if len(summaries) > 1:
        print_summary_table(summaries)

    if save_dir:
        print(f"Saved .prof files to: {os.path.abspath(save_dir)}")
        print("  Inspect with: python -m snakeviz profiles/<algo>.prof")
        print("  Or:           python -m pstats profiles/<algo>.prof")


if __name__ == "__main__":
    main()
