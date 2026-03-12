"""Shared comparison utilities for benchmarking algorithms on **discrete** problems.

Provides a reusable framework to run multiple algorithms on discrete
benchmark problems (TSP, Knapsack, Graph Coloring), collect convergence /
quality / time / robustness statistics, and generate comparison plots.

Supports optional **parameter tuning** via grid search before comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import itertools
import threading
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

# ── Ensure ``src/`` is on the import path ────────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import matplotlib.pyplot as plt

from AIP.problems.base_problem import DiscreteProblem
from AIP.problems.discrete.tsp import TSP
from AIP.problems.discrete.knapsack import Knapsack
from AIP.problems.discrete.graph_coloring import GraphColoring

# ── Algorithm imports ────────────────────────────────────────────────────
# Classical graph-search
from AIP.algorithm.classical.DFS import DepthFirstSearch
from AIP.algorithm.classical.BFS import BreadthFirstSearch
from AIP.algorithm.classical.UCS import UniformCostSearch
from AIP.algorithm.classical.GreedyBestFirst import GreedyBestFirstSearch
from AIP.algorithm.classical.AStar import AStarSearch
# Local search
from AIP.algorithm.local.HillClimbing import HillClimbing, HillClimbingParameter
# Natural-inspired
from AIP.algorithm.natural.physic.SA import SimulatedAnnealing, SimulatedAnnealingParameter
from AIP.algorithm.natural.physic.HS import HarmonySearch
from AIP.algorithm.natural.physic.GSA import GravitationalSearchAlgorithm, GravitationalSearchParameter
from AIP.algorithm.natural.biology.abc import ArtificialBeeColony, ABCParameter
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter
from AIP.algorithm.natural.biology.aco import (
    AntSystem, AntSystemParameter,
    ACS, ACSParameter,
    MMAS, MMASParameter,
)

# ── Algorithm categories ──────────────────────────────────────────────────
# Classical algorithms are deterministic graph-search; they only need one run
# and produce no iterative convergence curve.
_CLASSICAL_ALGOS = {"DFS", "BFS", "UCS", "Greedy", "A*"}
# Algorithms that need a random initial state passed to run()
_NEEDS_INITIAL_STATE = {"HC"}

# =====================================================================
# Config directory for persisting tuned parameters
# =====================================================================

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _config_path(problem_name: str) -> str:
    return os.path.join(_CONFIGS_DIR, f"{problem_name}.json")


def save_tuned_config(problem_name: str, algo_name: str, params: dict) -> str:
    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = _config_path(problem_name)
    existing: dict[str, dict] = {}
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing = json.load(f)
    existing[algo_name] = params
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    return os.path.abspath(path)


def save_all_tuned_configs(problem_name: str, tuned_params: dict[str, dict]) -> str:
    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = _config_path(problem_name)
    existing: dict[str, dict] = {}
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing = json.load(f)
    existing.update(tuned_params)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    return os.path.abspath(path)


def load_tuned_config(
    problem_name: str,
    algo_names: list[str] | None = None,
) -> dict[str, dict] | None:
    path = _config_path(problem_name)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data: dict[str, dict] = json.load(f)
    if algo_names is not None:
        data = {k: v for k, v in data.items() if k in algo_names}
    return data if data else None


# =====================================================================
# Parameter grids for tuning (discrete algorithms)
# =====================================================================

# --- Common grids (work for all discrete problems) ---
PARAM_GRIDS_COMMON: dict[str, dict[str, list]] = {
    "SA": {
        "initial_temperature": [100.0, 500.0, 1000.0],
        "cooling_rate": [0.99, 0.995, 0.999],
    },
    "HS": {
        "hms": [20, 50],
        "hmcr": [0.8, 0.9, 0.95],
        "par": [0.1, 0.3, 0.5],
        "bw": [0.05, 0.1],
    },
    "ABC": {
        "n_bees": [30, 50],
        "limit": [20, 50, 100],
    },
    "CS": {
        "n_nests": [25, 50],
        "pa": [0.15, 0.25, 0.35],
        "alpha": [0.01, 0.1, 0.5],
    },
    "FA": {
        "n_fireflies": [25, 50],
        "alpha": [0.2, 0.5],
        "beta0": [0.5, 1.0],
        "gamma": [0.5, 1.0],
        "alpha_decay": [0.95, 0.98],
    },
    "GSA": {
        "pop_size": [30, 50],
        "G0": [50.0, 100.0],
        "alpha": [10.0, 20.0],
    },
}

# --- TSP-only grids (ACO family) ---
PARAM_GRIDS_TSP: dict[str, dict[str, list]] = {
    "AS": {
        "rho": [0.3, 0.5],
        "m": [20, 30],
        "q": [50.0, 100.0],
        "alpha": [1.0],
        "beta": [2.0, 3.0],
    },
    "ACS": {
        "rho": [0.1, 0.2],
        "xi": [0.05, 0.1],
        "m": [20, 30],
        "q0": [0.8, 0.9],
        "alpha": [1.0],
        "beta": [2.0, 3.0],
    },
    "MMAS": {
        "rho": [0.02, 0.05],
        "m": [20, 30],
        "alpha": [1.0],
        "beta": [2.0, 3.0, 5.0],
    },
}


def get_param_grids(
    problem_type: Literal["TSP", "Knapsack", "GraphColoring"],
) -> dict[str, dict[str, list]]:
    """Return combined parameter grids for the given problem type."""
    grids = dict(PARAM_GRIDS_COMMON)
    if problem_type == "TSP":
        grids.update(PARAM_GRIDS_TSP)
    return grids


# =====================================================================
# Algorithm builder — creates an algorithm from name + param dict
# =====================================================================

def build_algo(
    algo_name: str,
    params: dict,
    problem: DiscreteProblem,
    cycle: int,
):
    """Construct a discrete algorithm instance from a name and parameter dict.

    Parameters
    ----------
    algo_name : str
        Algorithm key (e.g. ``'SA'``, ``'ABC'``, ``'AS'``).
    params : dict
        Hyperparameter values.
    problem : DiscreteProblem
        The benchmark problem.
    cycle : int
        Number of iterations.

    Returns
    -------
    model
        A ready-to-run algorithm instance.
    """
    match algo_name:
        # ── Classical graph-search (no hyperparameters) ────────────
        case "DFS":
            return DepthFirstSearch({}, problem)
        case "BFS":
            return BreadthFirstSearch({}, problem)
        case "UCS":
            return UniformCostSearch({}, problem)
        case "Greedy":
            return GreedyBestFirstSearch({}, problem)
        case "A*":
            return AStarSearch({}, problem)
        # ── Local search ─────────────────────────────────────────
        case "HC":
            cfg = HillClimbingParameter(
                iteration=params.get("iteration", cycle),
            )
            return HillClimbing(cfg, problem)
        case "SA":
            cfg = SimulatedAnnealingParameter(
                initial_temperature=params.get("initial_temperature", 500.0),
                cooling_rate=params.get("cooling_rate", 0.995),
                min_temperature=params.get("min_temperature", 0.01),
                max_iterations=cycle,
                n_flips=params.get("n_flips", 1),
            )
            return SimulatedAnnealing(cfg, problem)
        case "HS":
            return HarmonySearch(
                {
                    "hms": params.get("hms", 30),
                    "hmcr": params.get("hmcr", 0.9),
                    "par": params.get("par", 0.3),
                    "bw": params.get("bw", 0.1),
                    "max_iterations": params.get("max_iterations", cycle),
                },
                problem,
            )
        case "ABC":
            cfg = ABCParameter(
                n_bees=params.get("n_bees", 30),
                limit=params.get("limit", 50),
                iteration=cycle,
            )
            return ArtificialBeeColony(cfg, problem)
        case "CS":
            cfg = CuckooSearchParameter(
                n_nests=params.get("n_nests", 25),
                pa=params.get("pa", 0.25),
                alpha=params.get("alpha", 0.01),
                beta=params.get("beta", 1.5),
                iteration=cycle,
            )
            return CuckooSearch(cfg, problem)
        case "FA":
            cfg = FireflyParameter(
                n_fireflies=params.get("n_fireflies", 25),
                alpha=params.get("alpha", 0.5),
                beta0=params.get("beta0", 1.0),
                gamma=params.get("gamma", 1.0),
                alpha_decay=params.get("alpha_decay", 0.97),
                cycle=cycle,
            )
            return FireflyAlgorithm(cfg, problem)
        case "GSA":
            cfg = GravitationalSearchParameter(
                iteration=params.get("max_iterations", cycle),
                G0=params.get("G0", 100.0),
                alpha=params.get("alpha", 20.0),
                pop_size=params.get("pop_size", 30),
            )
            return GravitationalSearchAlgorithm(cfg, problem)
        # ACO family (TSP only)
        case "AS":
            cfg = AntSystemParameter(
                rho=params.get("rho", 0.5),
                m=params.get("m", 20),
                q=params.get("q", 100.0),
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 2.0),
                cycle=cycle,
            )
            return AntSystem(cfg, problem)
        case "ACS":
            cfg = ACSParameter(
                rho=params.get("rho", 0.1),
                xi=params.get("xi", 0.1),
                m=params.get("m", 20),
                q0=params.get("q0", 0.9),
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 2.0),
                cycle=cycle,
            )
            return ACS(cfg, problem)
        case "MMAS":
            cfg = MMASParameter(
                rho=params.get("rho", 0.02),
                m=params.get("m", 20),
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 3.0),
                cycle=cycle,
            )
            return MMAS(cfg, problem)
        case _:
            raise ValueError(f"Unknown discrete algorithm: {algo_name}")


# =====================================================================
# Parameter tuning via grid search
# =====================================================================

def tune_algorithm(
    algo_name: str,
    problem: DiscreteProblem,
    problem_config_name: str,
    cycle: int = 500,
    n_runs: int = 5,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Run a grid search to find the best parameters for one algorithm."""
    grids = get_param_grids(_problem_type(problem))
    if algo_name not in grids:
        print(f"  [SKIP] No parameter grid for {algo_name}")
        return {}

    grid = grids[algo_name]
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    print(f"\n{'=' * 65}")
    print(f"  Tuning {algo_name}  |  {total} configs × {n_runs} runs")
    print(f"  Problem: {problem._name}  |  Cycle: {cycle}")
    print(f"{'=' * 65}")

    best_mean = float("inf")
    best_params: dict = {}

    for idx, combo in enumerate(combinations, start=1):
        params = dict(zip(param_names, combo))
        fitnesses: list[float] = []

        for rid in range(n_runs):
            np.random.seed(seed + rid)
            random.seed(seed + rid)
            model = build_algo(algo_name, params, problem, cycle)
            if algo_name in _NEEDS_INITIAL_STATE:
                initial = problem.sample(1).flatten()
                model.run(initial_state=initial)  # type: ignore[call-arg]
            else:
                model.run()
            assert model.best_fitness is not None
            fitnesses.append(float(model.best_fitness))

        mean_f = float(np.mean(fitnesses))
        print(
            f"  [{idx:>{len(str(total))}}/{total}] "
            f"{params}  ->  Mean: {mean_f:.8e}"
        )

        if mean_f < best_mean:
            best_mean = mean_f
            best_params = params.copy()

    print(f"\n  Best for {algo_name}: {best_params}  (Mean: {best_mean:.8e})")

    if save:
        cfg_path = save_tuned_config(problem_config_name, algo_name, best_params)
        print(f"  Config saved to: {cfg_path}")

    print(f"{'=' * 65}\n")
    return best_params


def tune_all_algorithms(
    problem: DiscreteProblem,
    problem_config_name: str,
    cycle: int = 500,
    n_runs: int = 5,
    seed: int = 42,
    algo_names: list[str] | None = None,
    save: bool = True,
) -> dict[str, dict]:
    """Tune parameters for all (or selected) algorithms on a discrete problem."""
    grids = get_param_grids(_problem_type(problem))
    if algo_names is None:
        algo_names = list(grids.keys())

    tuned: dict[str, dict] = {}
    for name in algo_names:
        if name in grids:
            tuned[name] = tune_algorithm(
                name, problem, problem_config_name, cycle, n_runs, seed, save=save,
            )
        else:
            print(f"  [SKIP] No parameter grid for {name}")
            tuned[name] = {}

    if save:
        cfg_path = save_all_tuned_configs(problem_config_name, tuned)
        print(f"\n  All tuned configs saved to: {cfg_path}")

    return tuned


def _problem_type(problem: DiscreteProblem) -> Literal["TSP", "Knapsack", "GraphColoring"]:
    """Determine problem type string."""
    if isinstance(problem, TSP):
        return "TSP"
    elif isinstance(problem, Knapsack):
        return "Knapsack"
    elif isinstance(problem, GraphColoring):
        return "GraphColoring"
    raise ValueError(f"Unknown discrete problem type: {type(problem)}")


# =====================================================================
# Registry — default (name, builder_fn) pairs
# =====================================================================

ALGO_REGISTRY_CLASSICAL: dict[str, Callable] = {
    "DFS": lambda prob, cyc: build_algo("DFS", {}, prob, cyc),
    "BFS": lambda prob, cyc: build_algo("BFS", {}, prob, cyc),
    "UCS": lambda prob, cyc: build_algo("UCS", {}, prob, cyc),
    "Greedy": lambda prob, cyc: build_algo("Greedy", {}, prob, cyc),
    "A*": lambda prob, cyc: build_algo("A*", {}, prob, cyc),
}

ALGO_REGISTRY_COMMON: dict[str, Callable] = {
    **ALGO_REGISTRY_CLASSICAL,
    "HC": lambda prob, cyc: build_algo("HC", {}, prob, cyc),
    "SA": lambda prob, cyc: build_algo("SA", {}, prob, cyc),
    "HS": lambda prob, cyc: build_algo("HS", {}, prob, cyc),
    "ABC": lambda prob, cyc: build_algo("ABC", {}, prob, cyc),
    "CS": lambda prob, cyc: build_algo("CS", {}, prob, cyc),
    "FA": lambda prob, cyc: build_algo("FA", {}, prob, cyc),
    "GSA": lambda prob, cyc: build_algo("GSA", {}, prob, cyc),
}

ALGO_REGISTRY_TSP: dict[str, Callable] = {
    **ALGO_REGISTRY_COMMON,
    "AS": lambda prob, cyc: build_algo("AS", {}, prob, cyc),
    "ACS": lambda prob, cyc: build_algo("ACS", {}, prob, cyc),
    "MMAS": lambda prob, cyc: build_algo("MMAS", {}, prob, cyc),
}


def get_default_registry(
    problem_type: Literal["TSP", "Knapsack", "GraphColoring"],
) -> dict[str, Callable]:
    """Return the default algorithm registry for a problem type."""
    if problem_type == "TSP":
        return dict(ALGO_REGISTRY_TSP)
    return dict(ALGO_REGISTRY_COMMON)


def build_algo_registry(
    problem_type: Literal["TSP", "Knapsack", "GraphColoring"],
    tuned_params: dict[str, dict] | None = None,
) -> dict[str, Callable]:
    """Create an algorithm registry, optionally using tuned parameters."""
    base = get_default_registry(problem_type)
    if tuned_params is None:
        return base

    registry: dict[str, Callable] = {}
    for name in base:
        params = tuned_params.get(name, {})
        registry[name] = (
            lambda prob, cyc, _n=name, _p=params: build_algo(_n, _p, prob, cyc)
        )
    return registry


# =====================================================================
# Data collection
# =====================================================================

@dataclass
class RunResult:
    """Result of a single algorithm run."""
    algo_name: str
    run_id: int
    best_fitness: float
    time_ms: float
    fitness_curve: list[float]
    best_solution: object | None = None


# Algorithms whose history stores float fitness values directly
_FLOAT_HISTORY_ALGOS = {"HC"}


def _extract_fitness_curve(
    algo_name: str,
    model,
    problem: DiscreteProblem,
) -> list[float]:
    """Convert a model's history to a list of per-iteration best fitness."""
    # Classical graph-search algorithms have no iterative convergence curve;
    # their history stores explored *states*, not solution vectors.
    if algo_name in _CLASSICAL_ALGOS:
        bf = model.best_fitness
        return [float(bf) if bf is not None else float("inf")]

    history = model.history
    if not history:
        bf = model.best_fitness
        return [float(bf) if bf is not None else float("inf")]

    if algo_name in _FLOAT_HISTORY_ALGOS:
        return [float(v) for v in history]

    # History items are solution vectors → evaluate each
    curve: list[float] = []
    for sol in history:
        arr = np.asarray(sol)
        if arr.ndim == 0:
            curve.append(float(arr))
        else:
            curve.append(float(problem.eval(arr)))
    return curve


def run_comparison(
    problem: DiscreteProblem,
    cycle: int = 500,
    seed: int = 42,
    algo_names: list[str] | None = None,
    tuned_params: dict[str, dict] | None = None,
    skip_algos: set[str] | None = None,
    timeout: float = 60.0,
) -> list[RunResult]:
    """Run each algorithm once and collect results.

    Parameters
    ----------
    problem : DiscreteProblem
        The benchmark problem.
    cycle : int
        Number of iterations per run.
    seed : int
        Random seed for reproducibility.
    algo_names : list[str] or None
        Subset of algorithm names to run. ``None`` → run all.
    tuned_params : dict[str, dict] or None
        Optional tuned hyperparameter dict.
    skip_algos : set[str] or None
        Algorithm names to skip (e.g. classical algos on large instances).
    timeout : float
        Maximum seconds per algorithm run. If exceeded, the algorithm
        is skipped (default: 60).

    Returns
    -------
    list[RunResult]
    """
    ptype = _problem_type(problem)
    registry = build_algo_registry(ptype, tuned_params)

    if algo_names is None:
        algo_names = list(registry.keys())

    if skip_algos:
        algo_names = [n for n in algo_names if n not in skip_algos]

    results: list[RunResult] = []

    for name in algo_names:
        if name not in registry:
            print(f"  [SKIP] {name} not in registry for {ptype}")
            continue
        builder = registry[name]
        print(f"\n-- {name} (1 run, timeout={timeout:.0f}s) --")
        for rid in range(1, 2):
            np.random.seed(seed + rid)
            random.seed(seed + rid)

            model = builder(problem, cycle)

            # Run with timeout using a daemon thread
            exc_holder: list[BaseException] = []

            def _run_algo():
                try:
                    if name in _NEEDS_INITIAL_STATE:
                        initial = problem.sample(1).flatten()
                        model.run(initial_state=initial)  # type: ignore[call-arg]
                    else:
                        model.run()
                except Exception as e:
                    exc_holder.append(e)

            t0 = time.perf_counter()
            thread = threading.Thread(target=_run_algo, daemon=True)
            thread.start()
            thread.join(timeout=timeout)
            t1 = time.perf_counter()

            if thread.is_alive():
                print(f"  TIMEOUT after {timeout:.0f}s — skipping")
                continue

            if exc_holder:
                print(f"  ERROR: {exc_holder[0]} — skipping")
                continue

            # Classical algos may return best_fitness=None when no solution found
            bf_raw = model.best_fitness
            if bf_raw is None:
                print(f"  NO SOLUTION FOUND — skipping  "
                      f"Time={(t1 - t0) * 1000.0:.1f} ms")
                continue
            else:
                bf = float(bf_raw)

            curve = _extract_fitness_curve(name, model, problem)

            # Capture best solution vector
            sol = getattr(model, 'best_solution', None)

            results.append(RunResult(
                algo_name=name,
                run_id=rid,
                best_fitness=bf,
                time_ms=(t1 - t0) * 1000.0,
                fitness_curve=curve,
                best_solution=sol,
            ))
            if bf_raw is not None:
                print(f"  Fitness={bf:.8e}  Time={results[-1].time_ms:.1f} ms")

    return results


# =====================================================================
# Plotting helpers
# =====================================================================

_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
]


def _group_by_algo(results: list[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for r in results:
        grouped.setdefault(r.algo_name, []).append(r)
    return grouped


def plot_comparison(
    results: list[RunResult],
    problem_name: str,
    problem_desc: str = "",
    save_path: str | None = None,
    negate_fitness: bool = False,
    fitness_label: str = "Best Fitness",
) -> None:
    """Generate a 1×2 comparison figure for discrete problems.

    Sub-plots
    ---------
    1. **Fitness** — bar chart of fitness per algorithm.
    2. **Computational Time** — bar chart of execution time.

    Parameters
    ----------
    negate_fitness : bool
        If ``True``, negate fitness values for display (e.g. Knapsack
        stores ``-profit`` internally; negating shows positive profit).
    fitness_label : str
        Label for the fitness axis (e.g. ``'Conflicts'``, ``'Profit'``,
        ``'Distance'``).
    """
    grouped = _group_by_algo(results)
    algo_names = list(grouped.keys())
    n_algos = len(algo_names)
    colors = (_COLORS * ((n_algos // len(_COLORS)) + 1))[:n_algos]
    sign = -1.0 if negate_fitness else 1.0

    title_suffix = f"  ({problem_desc})" if problem_desc else ""

    # Derive separate save paths from save_path
    if save_path:
        base, ext = os.path.splitext(save_path)
        save_fitness = f"{base}_fitness{ext}"
        save_time = f"{base}_time{ext}"
    else:
        save_fitness = None
        save_time = None

    # ── 1. Best Fitness (separate window) ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.suptitle(
        f"{fitness_label} — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    means_f = [sign * np.mean([r.best_fitness for r in grouped[n]]) for n in algo_names]
    stds_f = [np.std([r.best_fitness for r in grouped[n]]) for n in algo_names]
    bars = ax1.bar(algo_names, means_f, yerr=stds_f, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(fitness_label, fontsize=11)
    ax1.set_title(fitness_label, fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_f):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    fig1.tight_layout(rect=(0, 0, 1, 0.95))

    if save_fitness:
        os.makedirs(os.path.dirname(save_fitness) or ".", exist_ok=True)
        fig1.savefig(save_fitness, dpi=150, bbox_inches="tight")
        print(f"\nFitness figure saved to: {save_fitness}")

    # ── 2. Computational Time (separate window) ──────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.suptitle(
        f"Computational Time — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    means_t = [np.mean([r.time_ms for r in grouped[n]]) for n in algo_names]
    stds_t = [np.std([r.time_ms for r in grouped[n]]) for n in algo_names]
    bars = ax2.bar(algo_names, means_t, yerr=stds_t, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Time (ms)", fontsize=11)
    ax2.set_title("Computational Time", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_t):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))

    if save_time:
        os.makedirs(os.path.dirname(save_time) or ".", exist_ok=True)
        fig2.savefig(save_time, dpi=150, bbox_inches="tight")
        print(f"\nTime figure saved to: {save_time}")

    if not save_path:
        plt.show()


# Classical algos produce a single best-fitness value (no iterative curve).
_SINGLE_POINT_ALGOS = _CLASSICAL_ALGOS


def plot_convergence(
    results: list[RunResult],
    problem_name: str,
    problem_desc: str = "",
    save_path: str | None = None,
    negate_fitness: bool = False,
    fitness_label: str = "Best Fitness",
) -> None:
    """Plot convergence curves for all algorithms.

    Classical graph-search algorithms (DFS, BFS, UCS, Greedy, A*) are
    shown as horizontal dashed reference lines (single best-fitness value).
    Local search (HC) and natural-inspired algorithms are shown as
    iterative convergence curves.

    Parameters
    ----------
    results : list[RunResult]
        Run results from ``run_comparison``.
    problem_name : str
        Display name for the problem.
    problem_desc : str
        Extra description (size, dimensions, …).
    save_path : str or None
        If given, save the figure to this path instead of showing.
    negate_fitness : bool
        If True, negate fitness values for display (e.g. Knapsack).
    fitness_label : str
        Y-axis label (e.g. 'Distance', 'Profit', 'Conflicts').
    """
    if not results:
        print("  [INFO] No algorithm results to plot convergence.")
        return

    grouped = _group_by_algo(results)
    algo_names = list(grouped.keys())
    n_algos = len(algo_names)
    colors = (_COLORS * ((n_algos // len(_COLORS)) + 1))[:n_algos]
    sign = -1.0 if negate_fitness else 1.0

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        f"Convergence Speed — {problem_name}"
        + (f"  ({problem_desc})" if problem_desc else ""),
        fontsize=16, fontweight="bold", y=0.98,
    )

    # Find the max iteration length across iterative algorithms for x-axis
    max_iter = 1
    for name in algo_names:
        if name not in _SINGLE_POINT_ALGOS:
            for r in grouped[name]:
                if len(r.fitness_curve) > max_iter:
                    max_iter = len(r.fitness_curve)

    for idx, name in enumerate(algo_names):
        curves = [r.fitness_curve for r in grouped[name]]
        curve_len = max(len(c) for c in curves)

        if name in _SINGLE_POINT_ALGOS:
            # Draw a horizontal dashed line across the full x-axis
            val = sign * curves[0][0]
            ax.axhline(y=val, label=name, color=colors[idx],
                        linewidth=1.4, linestyle="--", alpha=0.7)
        else:
            # Plot iterative convergence curve
            arr = np.full((len(curves), curve_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean_c = sign * np.nanmean(arr, axis=0)
            x = np.arange(1, curve_len + 1)
            ax.plot(x, mean_c, label=name, color=colors[idx], linewidth=1.6)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(fitness_label, fontsize=12)
    ax.set_title("All Algorithms", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nConvergence figure saved to: {save_path}")
    else:
        plt.show()


def print_summary_table(
    results: list[RunResult],
    negate_fitness: bool = False,
    fitness_label: str = "Fitness",
    format_solution: Callable[[object], str] | None = None,
) -> None:
    """Print a summary table to the terminal.

    Parameters
    ----------
    negate_fitness : bool
        If ``True``, negate fitness values for display.
    fitness_label : str
        Column header for the fitness value.
    format_solution : callable or None
        A function that takes a best_solution object and returns a
        human-readable string (e.g. path for TSP, colors for Graph
        Coloring, selected items for Knapsack).  If ``None``, the
        solution column is omitted.
    """
    grouped = _group_by_algo(results)
    algo_names = list(grouped.keys())
    sign = -1.0 if negate_fitness else 1.0

    header = (f"  {'Algorithm':<8s} | {fitness_label:>14s} "
              f"| {'Time(ms)':>14s}")
    sep = "  " + "-" * len(header.strip())

    print(f"\n{'=' * len(header.strip())}")
    print(header)
    print(sep)

    for name in algo_names:
        fits = [r.best_fitness for r in grouped[name]]
        display_fit = sign * fits[0]
        times = [r.time_ms for r in grouped[name]]
        print(
            f"  {name:<8s} | {display_fit:>14.6e} "
            f"| {times[0]:>14.1f}"
        )
        if format_solution is not None:
            sol = grouped[name][0].best_solution
            if sol is not None:
                print(f"           Solution: {format_solution(sol)}")

    print(f"{'=' * len(header.strip())}\n")
