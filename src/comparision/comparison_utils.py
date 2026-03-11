"""Shared comparison utilities for benchmarking optimization algorithms.

Provides a reusable framework to run multiple algorithms on a continuous
benchmark problem, collect convergence / quality / time / robustness
statistics, and generate publication-quality 2-D comparison plots.

Supports optional **parameter tuning** via grid search before comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ── Ensure ``src/`` is on the import path ────────────────────────────────
_SRC = os.path.join(os.path.dirname(__file__), os.pardir)
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from AIP.problems.continuous.continuous import ContinuousProblem

# ── Algorithm imports ────────────────────────────────────────────────────
from AIP.algorithm.natural.evolution.ga import (
    GeneticAlgorithm, GAParameter, SelectionMethod, CrossoverMethod,
)
from AIP.algorithm.natural.evolution.de import (
    DifferentialEvolution, DEParameter, MutationStrategy, DECrossoverType,
    VariableType,
)
from AIP.algorithm.natural.biology.pso import ParticleSwarmOptimization, PSOParameter
from AIP.algorithm.natural.biology.abc import ArtificialBeeColony, ABCParameter
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter
from AIP.algorithm.natural.physic.SA import SimulatedAnnealing
from AIP.algorithm.natural.physic.HS import HarmonySearch
from AIP.algorithm.natural.physic.GSA import GravitationalSearchAlgorithm, GravitationalSearchParameter
from AIP.algorithm.natural.human.ca import CA, CAConfig
from AIP.algorithm.natural.human.sfo import SFO, SFOConfig
from AIP.algorithm.natural.human.tlbo import TLBO, TLBOConfig
from AIP.algorithm.natural.evolution.es import (
    OnePlusOneES, OnePlusOneESParameter,
    SelfAdaptiveES, SelfAdaptiveESParameter,
    CMAES, CMAESParameter,
    MuRhoPlusLambdaES, MuRhoPlusLambdaESParameter,
)
from AIP.algorithm.local.HillClimbing import HillClimbing, HillClimbingParameter


# =====================================================================
# Config directory for persisting tuned parameters
# =====================================================================

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _config_path(problem_name: str) -> str:
    """Return the JSON config file path for a given problem name."""
    return os.path.join(_CONFIGS_DIR, f"{problem_name}.json")


def save_tuned_config(
    problem_name: str,
    algo_name: str,
    params: dict,
) -> str:
    """Persist tuned parameters for one algorithm on one problem.

    The config file is ``configs/<problem_name>.json``.  It is a JSON
    object mapping algorithm names to their best-parameter dicts.
    If the file already exists, only the entry for *algo_name* is
    updated; other algorithms are left untouched.

    Parameters
    ----------
    problem_name : str
        Problem key (e.g. ``'Ackley'``, ``'Sphere'``).
    algo_name : str
        Algorithm key (e.g. ``'GA'``, ``'PSO'``).
    params : dict
        Best hyperparameter values.

    Returns
    -------
    str
        Absolute path to the saved config file.
    """
    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = _config_path(problem_name)

    # Load existing config if present
    existing: dict[str, dict] = {}
    if os.path.isfile(path):
        with open(path, "r") as f:
            existing = json.load(f)

    existing[algo_name] = params

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    return os.path.abspath(path)


def save_all_tuned_configs(
    problem_name: str,
    tuned_params: dict[str, dict],
) -> str:
    """Persist tuned parameters for all algorithms on one problem.

    Parameters
    ----------
    problem_name : str
        Problem key.
    tuned_params : dict[str, dict]
        Mapping from AIP.algorithm name to best-parameter dict.

    Returns
    -------
    str
        Absolute path to the saved config file.
    """
    os.makedirs(_CONFIGS_DIR, exist_ok=True)
    path = _config_path(problem_name)

    # Load existing, then merge
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
    """Load previously saved tuned parameters for a problem.

    Parameters
    ----------
    problem_name : str
        Problem key (e.g. ``'Ackley'``).
    algo_names : list[str] or None
        If given, only return entries for these algorithms.

    Returns
    -------
    dict[str, dict] or None
        Mapping from AIP.algorithm name to parameter dict, or ``None``
        if the config file does not exist.
    """
    path = _config_path(problem_name)
    if not os.path.isfile(path):
        return None

    with open(path, "r") as f:
        data: dict[str, dict] = json.load(f)

    if algo_names is not None:
        data = {k: v for k, v in data.items() if k in algo_names}

    if not data:
        return None

    return data


# =====================================================================
# Parameter grids for tuning
# =====================================================================

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "GA": {
        "pop_size": [50, 100],
        "pc":       [0.7, 0.8, 0.9],
        "pm":       [0.01, 0.02, 0.05],
    },
    "DE": {
        "pop_size": [50, 100],
        "F":        [0.5, 0.8, 1.0],
        "Cr":       [0.7, 0.9],
    },
    "PSO": {
        "n_particles": [30, 50, 100],
        "w":           [0.4, 0.7, 0.9],
        "c1":          [1.5, 2.0],
        "c2":          [1.5, 2.0],
    },
    "ABC": {
        "n_bees": [30, 50, 100],
        "limit":  [20, 50, 100],
    },
    "CS": {
        "n_nests": [25, 50],
        "pa":      [0.15, 0.25, 0.35],
        "alpha":   [0.1, 0.5, 1.0],
    },
    "FA": {
        "n_fireflies": [25, 50],
        "alpha":       [0.2, 0.5],
        "beta0":       [0.5, 1.0],
        "gamma":       [0.5, 1.0],
        "alpha_decay":  [0.95, 0.98],
    },
    "SA": {
        "initial_temperature": [100.0, 1000.0],
        "cooling_rate":        [0.9, 0.95, 0.99],
        "step_size":           [0.05, 0.1, 0.5],
    },
    "HS": {
        "hms":  [20, 50],
        "hmcr": [0.8, 0.9, 0.95],
        "par":  [0.1, 0.3, 0.5],
        "bw":   [0.05, 0.1, 0.2],
    },
    "CA": {
        "pop_size":       [50, 100],
        "accepted_ratio": [0.1, 0.2, 0.3],
        "exploit_ratio":  [0.7, 0.8, 0.9],
        "explore_sigma":  [0.05, 0.1, 0.2],
    },
    "SFO": {
        "pop_size":  [50, 100],
        "w":         [0.7, 0.9],
        "c_attract": [1.0, 1.5, 2.0],
        "c_social":  [1.0, 1.5, 2.0],
    },
    "TLBO": {
        "pop_size": [50, 100, 150],
    },
    "(1+1)-ES": {
        "sigma": [0.1, 0.5, 1.0],
    },
    "SA-ES": {
        "mu": [10, 20],
        "lam": [50, 100],
        "rho": [2, 5],
        "sigma_init": [0.1, 0.5],
    },
    "CMA-ES": {
        "sigma": [0.3, 0.5, 1.0],
    },
    "(μ/ρ+λ)-ES": {
        "mu": [10, 20],
        "rho": [2, 5],
        "lam": [50, 100],
        "sigma_init": [0.1, 0.5],
    },
    "GSA": {
        "pop_size": [30, 50],
        "G0": [50.0, 100.0],
        "alpha": [10.0, 20.0],
    },
    "HC": {
        "iteration": [500, 1000, 2000],
    },
}


# =====================================================================
# Algorithm builder — creates an algorithm from name + param dict
# =====================================================================

def build_algo(
    algo_name: str,
    params: dict,
    problem: ContinuousProblem,
    cycle: int,
):
    """Construct an algorithm instance from a name and parameter dict.

    Parameters
    ----------
    algo_name : str
        Algorithm key (e.g. ``'GA'``, ``'PSO'``).
    params : dict
        Hyperparameter values.  Missing keys fall back to defaults.
    problem : ContinuousProblem
        The benchmark problem.
    cycle : int
        Number of iterations.

    Returns
    -------
    model
        A ready-to-run algorithm instance.
    """
    match algo_name:
        case "GA":
            cfg = GAParameter(
                pop_size=params.get("pop_size", 50),
                n_bits=params.get("n_bits", 16),
                pc=params.get("pc", 0.8),
                pm=params.get("pm", 0.02),
                cycle=cycle,
                selection=params.get("selection", SelectionMethod.STOCHASTIC_REMAINDER),
                crossover=params.get("crossover", CrossoverMethod.TWO_SITE),
            )
            return GeneticAlgorithm(cfg, problem)
        case "DE":
            cfg = DEParameter(
                pop_size=params.get("pop_size", 50),
                F=params.get("F", 0.8),
                Cr=params.get("Cr", 0.9),
                cycle=cycle,
                strategy=params.get("strategy", MutationStrategy.RAND_1),
                crossover_type=params.get("crossover_type", DECrossoverType.BIN),
                variable_type=params.get("variable_type", VariableType.CONTINUOUS),
            )
            return DifferentialEvolution(cfg, problem)
        case "PSO":
            cfg = PSOParameter(
                n_particles=params.get("n_particles", 50),
                w=params.get("w", 0.7),
                c1=params.get("c1", 1.5),
                c2=params.get("c2", 1.5),
                v_max=params.get("v_max", None),
                cycle=cycle,
            )
            return ParticleSwarmOptimization(cfg, problem)
        case "ABC":
            n_bees = 50
            cfg = ABCParameter(
                n_bees=params.get("n_bees", n_bees),
                limit=params.get("limit", n_bees * problem.n_dim / 2),
                iteration=cycle,
            )
            return ArtificialBeeColony(cfg, problem)
        case "CS":
            cfg = CuckooSearchParameter(
                n_nests=params.get("n_nests", 40),
                pa=params.get("pa", 0.25),
                alpha=params.get("alpha", 1),
                beta=params.get("beta", 1.5),
                iteration=cycle,
            )
            return CuckooSearch(cfg, problem)
        case "FA":
            cfg = FireflyParameter(
                n_fireflies=params.get("n_fireflies", 50),
                alpha=params.get("alpha", 0.3),
                beta0=params.get("beta0", 1.0),
                gamma=params.get("gamma", 1.0),
                alpha_decay=params.get("alpha_decay", 0.97),
                cycle=cycle,
            )
            return FireflyAlgorithm(cfg, problem)
        case "SA":
            return SimulatedAnnealing(
                {
                    "initial_temperature": params.get("initial_temperature", 100.0),
                    "cooling_rate": params.get("cooling_rate", 0.95),
                    "min_temperature": params.get("min_temperature", 1e-8),
                    "max_iterations": cycle,
                    "step_size": params.get("step_size", 0.1),
                },
                problem,
            )
        case "HS":
            return HarmonySearch(
                {
                    "hms": params.get("hms", 50),
                    "hmcr": params.get("hmcr", 0.9),
                    "par": params.get("par", 0.3),
                    "bw": params.get("bw", 0.1),
                    "max_iterations": cycle,
                },
                problem,
            )
        case "CA":
            cfg = CAConfig(
                pop_size=params.get("pop_size", 50),
                iterations=cycle,
                minimization=True,
                accepted_ratio=params.get("accepted_ratio", 0.2),
                exploit_ratio=params.get("exploit_ratio", 0.8),
                explore_sigma=params.get("explore_sigma", 0.1),
            )
            return CA(cfg, problem)
        case "SFO":
            cfg = SFOConfig(
                pop_size=params.get("pop_size", 50),
                iterations=cycle,
                minimization=True,
                w=params.get("w", 0.9),
                w_decay=params.get("w_decay", 0.99),
                c_attract=params.get("c_attract", 1.5),
                c_social=params.get("c_social", 1.5),
            )
            return SFO(cfg, problem)
        case "TLBO":
            cfg = TLBOConfig(
                pop_size=params.get("pop_size", 50),
                iterations=cycle,
                minimization=True,
            )
            return TLBO(cfg, problem)
        case "(1+1)-ES":
            cfg = OnePlusOneESParameter(
                sigma=params.get("sigma", 0.5),
                cycle=cycle,
            )
            return OnePlusOneES(cfg, problem)
        case "SA-ES":
            cfg = SelfAdaptiveESParameter(
                mu=params.get("mu", 15),
                lam=params.get("lam", 100),
                rho=params.get("rho", 2),
                sigma_init=params.get("sigma_init", 0.5),
                cycle=cycle,
            )
            return SelfAdaptiveES(cfg, problem)
        case "CMA-ES":
            cfg = CMAESParameter(
                sigma=params.get("sigma", 0.5),
                cycle=cycle,
            )
            return CMAES(cfg, problem)
        case "(μ/ρ+λ)-ES":
            cfg = MuRhoPlusLambdaESParameter(
                mu=params.get("mu", 15),
                rho=params.get("rho", 2),
                lam=params.get("lam", 100),
                sigma_init=params.get("sigma_init", 0.5),
                cycle=cycle,
            )
            return MuRhoPlusLambdaES(cfg, problem)
        case "GSA":
            cfg = GravitationalSearchParameter(
                iteration=cycle,
                G0=params.get("G0", 100.0),
                alpha=params.get("alpha", 20.0),
                pop_size=params.get("pop_size", 30),
            )
            return GravitationalSearchAlgorithm(cfg, problem)
        case "HC":
            cfg = HillClimbingParameter(
                iteration=params.get("iteration", cycle),
            )
            return HillClimbing(cfg, problem)
        case _:
            raise ValueError(f"Unknown algorithm: {algo_name}")


# =====================================================================
# Parameter tuning via grid search
# =====================================================================

def tune_algorithm(
    algo_name: str,
    problem: ContinuousProblem,
    cycle: int = 200,
    n_runs: int = 5,
    seed: int = 42,
    save: bool = True,
) -> dict:
    """Run a grid search to find the best parameters for one algorithm.

    After tuning, the result is automatically saved to
    ``configs/<problem_name>.json`` unless *save* is ``False``.

    Parameters
    ----------
    algo_name : str
        Algorithm key (must exist in ``PARAM_GRIDS``).
    problem : ContinuousProblem
        The benchmark problem.
    cycle : int
        Iterations per run during tuning.
    n_runs : int
        Independent runs per parameter combination.
    seed : int
        Base random seed.
    save : bool
        Persist the best parameters to the config file (default True).

    Returns
    -------
    dict
        Best hyperparameter dict found.
    """
    grid = PARAM_GRIDS[algo_name]
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
            model.run()
            assert model.best_fitness is not None
            fitnesses.append(model.best_fitness)

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
        cfg_path = save_tuned_config(problem._name, algo_name, best_params)
        print(f"  Config saved to: {cfg_path}")

    print(f"{'=' * 65}\n")
    return best_params


def tune_all_algorithms(
    problem: ContinuousProblem,
    cycle: int = 200,
    n_runs: int = 5,
    seed: int = 42,
    algo_names: list[str] | None = None,
    save: bool = True,
) -> dict[str, dict]:
    """Tune parameters for all (or selected) algorithms.

    After tuning, all results are saved to ``configs/<problem_name>.json``
    unless *save* is ``False``.

    Parameters
    ----------
    problem : ContinuousProblem
        The benchmark problem.
    cycle : int
        Iterations per tuning run.
    n_runs : int
        Independent runs per config.
    seed : int
        Base random seed.
    algo_names : list[str] or None
        Algorithms to tune.  ``None`` → tune all in ``PARAM_GRIDS``.
    save : bool
        Persist the best parameters to the config file (default True).

    Returns
    -------
    dict[str, dict]
        Mapping from AIP.algorithm name to best parameter dict.
    """
    if algo_names is None:
        algo_names = list(PARAM_GRIDS.keys())

    tuned: dict[str, dict] = {}
    for name in algo_names:
        if name in PARAM_GRIDS:
            tuned[name] = tune_algorithm(
                name, problem, cycle, n_runs, seed, save=save,
            )
        else:
            print(f"  [SKIP] No parameter grid for {name}")
            tuned[name] = {}

    if save:
        cfg_path = save_all_tuned_configs(problem._name, tuned)
        print(f"\n  All tuned configs saved to: {cfg_path}")

    return tuned


# =====================================================================
# Registry — default (name, builder_fn) pairs
# =====================================================================
# Each builder receives ``(problem, cycle)`` and returns a ready model.
# These use sensible defaults.  Use ``build_algo_registry`` to create
# a registry from tuned parameters.

ALGO_REGISTRY: dict[str, Callable] = {
    "GA": lambda prob, cyc: build_algo("GA", {}, prob, cyc),
    "DE": lambda prob, cyc: build_algo("DE", {}, prob, cyc),
    "PSO": lambda prob, cyc: build_algo("PSO", {}, prob, cyc),
    "ABC": lambda prob, cyc: build_algo("ABC", {}, prob, cyc),
    "CS": lambda prob, cyc: build_algo("CS", {}, prob, cyc),
    "FA": lambda prob, cyc: build_algo("FA", {}, prob, cyc),
    "SA": lambda prob, cyc: build_algo("SA", {}, prob, cyc),
    "HS": lambda prob, cyc: build_algo("HS", {}, prob, cyc),
    "CA": lambda prob, cyc: build_algo("CA", {}, prob, cyc),
    "SFO": lambda prob, cyc: build_algo("SFO", {}, prob, cyc),
    "TLBO": lambda prob, cyc: build_algo("TLBO", {}, prob, cyc),
    "(1+1)-ES": lambda prob, cyc: build_algo("(1+1)-ES", {}, prob, cyc),
    "SA-ES": lambda prob, cyc: build_algo("SA-ES", {}, prob, cyc),
    "CMA-ES": lambda prob, cyc: build_algo("CMA-ES", {}, prob, cyc),
    "(μ/ρ+λ)-ES": lambda prob, cyc: build_algo("(μ/ρ+λ)-ES", {}, prob, cyc),
    "GSA": lambda prob, cyc: build_algo("GSA", {}, prob, cyc),
    "HC": lambda prob, cyc: build_algo("HC", {}, prob, cyc),
}


def build_algo_registry(
    tuned_params: dict[str, dict] | None = None,
) -> dict[str, Callable]:
    """Create an algorithm registry that uses tuned parameters.

    Parameters
    ----------
    tuned_params : dict[str, dict] or None
        Mapping from AIP.algorithm name to best-parameter dict.  Algorithms
        not present fall back to defaults.

    Returns
    -------
    dict[str, callable]
        Registry mapping name → ``lambda prob, cycle: model``.
    """
    if tuned_params is None:
        return dict(ALGO_REGISTRY)

    registry: dict[str, Callable] = {}
    for name in ALGO_REGISTRY:
        params = tuned_params.get(name, {})
        # Capture ``name`` and ``params`` in the closure correctly
        registry[name] = (
            lambda prob, cyc, _n=name, _p=params: build_algo(_n, _p, prob, cyc)
        )
    return registry

# Algorithms whose history stores float fitness values directly
_FLOAT_HISTORY_ALGOS = {
    "CA", "SFO", "TLBO",
    "(1+1)-ES", "SA-ES", "CMA-ES", "(μ/ρ+λ)-ES",
    "HC",
}


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
    fitness_curve: list[float]      # per-iteration best fitness


def _extract_fitness_curve(
    algo_name: str,
    model,
    problem: ContinuousProblem,
) -> list[float]:
    """Convert a model's history to a list of per-iteration best fitness.

    Some algorithms store solution vectors; others store floats directly.
    """
    history = model.history
    if not history:
        return [float(model.best_fitness)]

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
    problem: ContinuousProblem,
    cycle: int = 200,
    n_runs: int = 30,
    seed: int = 42,
    algo_names: list[str] | None = None,
    tuned_params: dict[str, dict] | None = None,
) -> list[RunResult]:
    """Run selected algorithms multiple times and collect results.

    Parameters
    ----------
    problem : ContinuousProblem
        The benchmark problem (2-D recommended for visualisation).
    cycle : int
        Number of iterations per run.
    n_runs : int
        How many independent runs per algorithm.
    seed : int
        Base random seed for reproducibility.
    algo_names : list[str] or None
        Subset of algorithm names to run. ``None`` → run all.
    tuned_params : dict[str, dict] or None
        Optional mapping from AIP.algorithm name to tuned hyperparameter dict.
        When provided, algorithms are built using ``build_algo`` with the
        tuned parameters instead of the default registry.

    Returns
    -------
    list[RunResult]
    """
    registry = build_algo_registry(tuned_params)

    if algo_names is None:
        algo_names = list(registry.keys())

    results: list[RunResult] = []

    for name in algo_names:
        builder = registry[name]
        print(f"\n-- {name} ({n_runs} runs) --")
        for rid in range(1, n_runs + 1):
            np.random.seed(seed + rid)
            random.seed(seed + rid)

            model = builder(problem, cycle)

            t0 = time.perf_counter()
            model.run()
            t1 = time.perf_counter()

            curve = _extract_fitness_curve(name, model, problem)
            bf = float(model.best_fitness)

            results.append(RunResult(
                algo_name=name,
                run_id=rid,
                best_fitness=bf,
                time_ms=(t1 - t0) * 1000.0,
                fitness_curve=curve,
            ))
            print(f"  Run {rid:>{len(str(n_runs))}}/{n_runs}  "
                  f"Fitness={bf:.8e}  Time={results[-1].time_ms:.1f} ms")

    return results


# =====================================================================
# Plotting helpers
# =====================================================================

_COLORS = [
    "#D32F2F",  # dark red
    "#1976D2",  # dark blue
    "#388E3C",  # dark green
    "#F57C00",  # dark orange
    "#7B1FA2",  # dark purple
    "#00838F",  # dark cyan
    "#C2185B",  # dark pink / rose
    "#5D4037",  # brown
    "#455A64",  # blue-grey
    "#AFB42B",  # dark lime / olive-yellow
    "#0097A7",  # teal
    "#E64A19",  # deep orange
    "#512DA8",  # deep purple
    "#00695C",  # dark teal
    "#AD1457",  # dark magenta
    "#283593",  # indigo
    "#827717",  # dark olive
]

# Markers placed sparsely on convergence curves for extra distinction
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h",
            "<", ">", "d", "H", "8", "+", "x"]


def _group_by_algo(results: list[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for r in results:
        grouped.setdefault(r.algo_name, []).append(r)
    return grouped


def plot_comparison(
    results: list[RunResult],
    problem_name: str,
    n_dim: int = 2,
    save_path: str | None = None,
) -> None:
    """Generate a 2×2 comparison figure.

    Sub-plots
    ---------
    1. **Convergence Speed** — mean best fitness vs iteration (log scale).
    2. **Solution Quality** — box-plot of final best fitness.
    3. **Computational Time** — bar chart of mean ± std execution time.
    4. **Robustness** — bar chart of std-dev of final fitness + success rate.

    Parameters
    ----------
    results : list[RunResult]
        Output of :func:`run_comparison`.
    problem_name : str
        Name shown in titles (e.g. "Ackley").
    n_dim : int
        Problem dimensionality (for titles).
    save_path : str or None
        If given, save the figure to this path instead of showing it.
    """
    grouped = _group_by_algo(results)
    algo_names = list(grouped.keys())
    n_algos = len(algo_names)
    colors = (_COLORS * ((n_algos // len(_COLORS)) + 1))[:n_algos]

    title_suffix = f" (dim = {n_dim})"

    # Derive separate save paths from save_path
    if save_path:
        base, ext = os.path.splitext(save_path)
        save_convergence = f"{base}_convergence{ext}"
        save_quality = f"{base}_quality{ext}"
        save_time = f"{base}_time{ext}"
        save_robustness = f"{base}_robustness{ext}"
    else:
        save_convergence = save_quality = save_time = save_robustness = None

    # ── 1. Convergence Speed (separate window) ───────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.suptitle(
        f"Convergence Speed — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    for idx, name in enumerate(algo_names):
        curves = [r.fitness_curve for r in grouped[name]]
        max_len = max(len(c) for c in curves)
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        mean_c = np.nanmean(arr, axis=0)
        std_c = np.nanstd(arr, axis=0)
        x = np.arange(1, max_len + 1)
        marker = _MARKERS[idx % len(_MARKERS)]
        # Place markers sparsely (every ~10% of iterations)
        mark_every = max(1, max_len // 10)
        ax1.plot(x, mean_c, label=name, color=colors[idx],
                 linewidth=1.6,
                 marker=marker, markersize=4, markevery=mark_every)
        ax1.fill_between(x, mean_c - std_c, mean_c + std_c,
                         color=colors[idx], alpha=0.08)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best Fitness (log)", fontsize=11)
    ax1.set_title("Convergence Speed", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both", linestyle="--")
    fig1.tight_layout(rect=(0, 0, 1, 0.95))

    if save_convergence:
        os.makedirs(os.path.dirname(save_convergence) or ".", exist_ok=True)
        fig1.savefig(save_convergence, dpi=150, bbox_inches="tight")
        print(f"\nConvergence figure saved to: {save_convergence}")

    # ── 2. Solution Quality (separate window) ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.suptitle(
        f"Solution Quality — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    box_data = [
        [r.best_fitness for r in grouped[name]] for name in algo_names
    ]
    bp = ax2.boxplot(
        box_data, tick_labels=algo_names, patch_artist=True, widths=0.6,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.set_ylabel("Best Fitness", fontsize=11)
    ax2.set_title("Solution Quality", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))

    if save_quality:
        os.makedirs(os.path.dirname(save_quality) or ".", exist_ok=True)
        fig2.savefig(save_quality, dpi=150, bbox_inches="tight")
        print(f"\nQuality figure saved to: {save_quality}")

    # ── 3. Computational Time (separate window) ──────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fig3.suptitle(
        f"Computational Time — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    means_t = [np.mean([r.time_ms for r in grouped[n]]) for n in algo_names]
    stds_t = [np.std([r.time_ms for r in grouped[n]]) for n in algo_names]
    bars = ax3.bar(algo_names, means_t, yerr=stds_t, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Time (ms)", fontsize=11)
    ax3.set_title("Computational Time", fontsize=13, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_t):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig3.tight_layout(rect=(0, 0, 1, 0.95))

    if save_time:
        os.makedirs(os.path.dirname(save_time) or ".", exist_ok=True)
        fig3.savefig(save_time, dpi=150, bbox_inches="tight")
        print(f"\nTime figure saved to: {save_time}")

    # ── 4. Robustness (separate window) ──────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    fig4.suptitle(
        f"Robustness — {problem_name}{title_suffix}",
        fontsize=16, fontweight="bold", y=0.98,
    )
    stds_f = [np.std([r.best_fitness for r in grouped[n]]) for n in algo_names]
    x_pos = np.arange(n_algos)
    width = 0.4
    bars1 = ax4.bar(x_pos - width / 2, stds_f, width, label="Std Dev (fitness)",
                    color=colors, alpha=0.6, edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Std Dev of Best Fitness", fontsize=11)
    ax4.set_title("Robustness", fontsize=13, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(algo_names, rotation=45, fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y", linestyle="--")

    ax4r = ax4.twinx()
    means_f = [np.mean([r.best_fitness for r in grouped[n]]) for n in algo_names]
    cv = [s / abs(m) if abs(m) > 1e-30 else 0.0
          for s, m in zip(stds_f, means_f)]
    ax4r.plot(x_pos, cv, "D-", color="red", markersize=5, linewidth=1.2,
              label="CV (Std/Mean)")
    ax4r.set_ylabel("Coefficient of Variation", fontsize=10, color="red")
    ax4r.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig4.tight_layout(rect=(0, 0, 1, 0.95))

    if save_robustness:
        os.makedirs(os.path.dirname(save_robustness) or ".", exist_ok=True)
        fig4.savefig(save_robustness, dpi=150, bbox_inches="tight")
        print(f"\nRobustness figure saved to: {save_robustness}")

    if not save_path:
        plt.show()


def print_summary_table(results: list[RunResult]) -> None:
    """Print a summary table to the terminal."""
    grouped = _group_by_algo(results)
    algo_names = list(grouped.keys())

    header = (f"  {'Algorithm':<8s} | {'Mean Fitness':>14s} | {'Std Fitness':>14s} "
              f"| {'Best':>14s} | {'Worst':>14s} | {'Mean Time(ms)':>14s}")
    sep = "  " + "-" * len(header.strip())

    print(f"\n{'=' * len(header.strip())}")
    print(header)
    print(sep)

    for name in algo_names:
        fits = [r.best_fitness for r in grouped[name]]
        times = [r.time_ms for r in grouped[name]]
        print(
            f"  {name:<8s} | {np.mean(fits):>14.6e} | {np.std(fits):>14.6e} "
            f"| {np.min(fits):>14.6e} | {np.max(fits):>14.6e} "
            f"| {np.mean(times):>14.1f}"
        )

    print(f"{'=' * len(header.strip())}\n")
