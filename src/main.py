"""Entry-point script: run Social Force Optimization on the Rastrigin function.

Usage
-----
    python main.py [--seed SEED] (seed is optional, default: 42)

The script
  1. Instantiates the n-dimensional benchmark problem.
  2. Configures and runs the algorithm.
  3. Prints the best fitness and solution found.
  4. Displays a professional convergence plot on a log-scaled y-axis.
"""

import random
import argparse
import itertools
import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from problems import Sphere, Rastrigin, Ackley, Cigar, Ridge
from algorithm import (
    GeneticAlgorithm,
    GAParameter,
    SelectionMethod,
    CrossoverMethod,
    DifferentialEvolution,
    DEParameter,
    MutationStrategy,
    DECrossoverType,
    EvolutionStrategy,
    ESVariant,
    OnePlusOneESParameter,
    SelfAdaptiveESParameter,
    CMAESParameter,
    MuRhoPlusLambdaESParameter,
    ParticleSwarmOptimization,
    PSOParameter,
    ABCParameter,
    ArtificialBeeColony,
    CuckooSearch,
    CuckooSearchParameter,
    FireflyAlgorithm,
    FireflyParameter
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def run_grid_search(
    target_algo: str,
    problem,
    n_dim: int,
    cycle: int,
    n_runs: int,
) -> None:
    """Execute a grid search over hyperparameters for a given algorithm.

    For each combination in the parameter grid the algorithm is executed
    ``n_runs`` times independently.  The combination that yields the
    lowest *mean best fitness* (minimization) is reported.

    Parameters
    ----------
    target_algo : str
        One of ``'CA'``, ``'SFO'``, ``'TLBO'``.
    problem : ContinuousProblem
        The benchmark problem instance to optimise.
    n_dim : int
        Dimensionality of the search space.
    cycle : int
        Number of iterations / generations per run.
    n_runs : int
        Independent repetitions for each parameter combination to
        ensure statistical robustness.
    """

    # ------------------------------------------------------------------
    # 1. Define the parameter grids per algorithm
    # ------------------------------------------------------------------
    grids: dict[str, dict[str, list]] = {
        "CA": {
            "pop_size":      [50, 100],
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
    }

    # Mapping from algorithm name to (ConfigClass, AlgorithmClass)
    algo_map: dict[str, tuple[type, type]] = {
        "CA":   (CAConfig, CA),
        "SFO":  (SFOConfig, SFO),
        "TLBO": (TLBOConfig, TLBO),
    }

    grid = grids[target_algo]
    ConfigCls, AlgoCls = algo_map[target_algo]

    # Build all parameter combinations using itertools.product
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    print(f"\n{'=' * 65}")
    print(f"  Grid Search for {target_algo}  |  {total} configs × {n_runs} runs")
    print(f"  Problem: {problem._name}  |  Dimensions: {n_dim}  |  Iterations: {cycle}")
    print(f"{'=' * 65}\n")

    best_mean_fitness = float("inf")
    best_config_dict: dict = {}

    # ------------------------------------------------------------------
    # 2. Iterate over every combination
    # ------------------------------------------------------------------
    for idx, combo in enumerate(combinations, start=1):
        config_dict = dict(zip(param_names, combo))

        # Build the config object with the current grid values + fixed params
        cfg = ConfigCls(
            iterations=cycle,
            minimization=True,
            **config_dict,
        )

        run_fitnesses: list[float] = []

        for run_id in range(n_runs):
            model = AlgoCls(configuration=cfg, problem=problem)
            model.run()
            run_fitnesses.append(float(model.best_fitness))

        mean_fitness = float(np.mean(run_fitnesses))

        print(
            f"  [{idx:>{len(str(total))}}/{total}] "
            f"{config_dict}  ->  Mean Fitness: {mean_fitness:.10f}"
        )

        if mean_fitness < best_mean_fitness:
            best_mean_fitness = mean_fitness
            best_config_dict = config_dict.copy()

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  GRID SEARCH RESULTS  –  {target_algo}")
    print(f"{'-' * 65}")
    print(f"  Best Mean Fitness : {best_mean_fitness:.10f}")
    print(f"  Best Configuration:")
    for k, v in best_config_dict.items():
        print(f"    {k:<18s} = {v}")
    print(f"{'=' * 65}\n")

def set_seed(seed: int | float | None):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
        print(f"Random seed set to: {seed}")


def run_ga(problem, n_dim: int, cycle: int):
    """Configure and run the Genetic Algorithm."""
    config = GAParameter(
        pop_size=100,
        n_bits=16,
        pc=0.8,
        pm=0.01,
        cycle=cycle,
        selection=SelectionMethod.STOCHASTIC_REMAINDER,
        crossover=CrossoverMethod.TWO_SITE,
    )
    ga = GeneticAlgorithm(configuration=config, problem=problem)

    print(f"\nRunning {ga.name}...")
    print(
        f"  pop_size={config.pop_size}, n_bits={config.n_bits}, "
        f"pc={config.pc}, pm={config.pm}, "
        f"selection={config.selection.value}, "
        f"crossover={config.crossover.value}"
    )

    best = ga.run()

    print(f"  Best Fitness : {ga.best_fitness:.10f}")
    print(f"  Best Solution: {best[:min(5, n_dim)]}{'...' if n_dim > 5 else ''}")

    return ga


def run_de(problem, n_dim: int, cycle: int):
    """Configure and run Differential Evolution."""
    config = DEParameter(
        pop_size=50,
        F=0.5,
        Cr=0.9,
        cycle=cycle,
        strategy=MutationStrategy.RAND_1,
        crossover_type=DECrossoverType.BIN,
    )
    de = DifferentialEvolution(configuration=config, problem=problem)

    print(
        f"\nRunning {de.name} (DE/{config.strategy.value}/{config.crossover_type.value})..."
    )
    print(f"  pop_size={config.pop_size}, F={config.F}, Cr={config.Cr}")

    best = de.run()

    print(f"  Best Fitness : {de.best_fitness:.10f}")
    print(f"  Best Solution: {best[:min(5, n_dim)]}{'...' if n_dim > 5 else ''}")

    return de


def run_es(problem, n_dim: int, cycle: int, variant: ESVariant) -> EvolutionStrategy:
    """Configure and run an Evolution Strategy variant via the façade."""
    match variant:
        case ESVariant.ONE_PLUS_ONE:
            config = OnePlusOneESParameter(
                sigma=1.0,
                cycle=cycle,
                adaptation_interval=0,  # auto → 10 * n_dim
                a=1.5,
            )
        case ESVariant.SELF_ADAPTIVE:
            config = SelfAdaptiveESParameter(
                mu=15,
                lam=100,
                rho=2,
                sigma_init=1.0,
                cycle=cycle,
            )
        case ESVariant.CMA_ES:
            config = CMAESParameter(
                sigma=1.0,
                cycle=cycle,
            )
        case ESVariant.MU_RHO_PLUS_LAMBDA:
            config = MuRhoPlusLambdaESParameter(
                mu=15,
                rho=5,
                lam=100,
                sigma_init=1.0,
                cycle=cycle,
            )

    es = EvolutionStrategy(variant=variant, configuration=config, problem=problem)

    print(f"\nRunning {es.name}...")
    print(f"  config: {config}")

    best = es.run()

    print(f"  Best Fitness : {es.best_fitness:.10f}")
    print(f"  Best Solution: {best[:min(5, n_dim)]}{'...' if n_dim > 5 else ''}")

    return es


def main():
    parser = argparse.ArgumentParser(
        description="Test GA, DE, and ES on benchmark problems"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Problem dimensionality (default: 10)",
    )
    parser.add_argument(
        "--cycle",
        type=int,
        default=500,
        help="Number of generations (default: 500)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="sphere",
        choices=["sphere", "rastrigin", "ackley", "cigar", "ridge"],
        help="Benchmark problem (default: ackley)",
    )
    parser.add_argument(
        "--tune", type=str, default="NONE",
        choices=["CA", "SFO", "TLBO", "NONE"],
        help="Run grid-search parameter tuning for the specified algorithm "
             "instead of the normal comparison (default: NONE)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of independent runs per grid-search config (default: 5)",
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

    # ── Grid Search mode ─────────────────────────────────────────────
    if args.tune != "NONE":
        run_grid_search(
            target_algo=args.tune,
            problem=problem,
            n_dim=args.dim,
            cycle=args.cycle,
            n_runs=args.runs,
        )
        return

    # ── Run all algorithms ───────────────────────────────────────────
    ga = run_ga(problem, args.dim, args.cycle)
    de = run_de(problem, args.dim, args.cycle)

    es_variants = [
        ESVariant.ONE_PLUS_ONE,
        ESVariant.SELF_ADAPTIVE,
        ESVariant.CMA_ES,
        ESVariant.MU_RHO_PLUS_LAMBDA,
    ]
    es_results: list[EvolutionStrategy] = []
    for variant in es_variants:
        es_results.append(run_es(problem, args.dim, args.cycle, variant))

    abc_config = ABCParameter(n_bees=50,limit=50 * args.dim/2,cycle=args.cycle)
    abc = ArtificialBeeColony(abc_config,problem)
    abc.run()

    # cs_config = CuckooSearchParameter(n_nests=50, pa=0.25, alpha=1,beta=1.5,cycle=args.cycle)
    # cs = CuckooSearch(cs_config, problem)
    fa_config = FireflyParameter(n_fireflies=25,alpha=0.25,beta0=0.001,gamma=1,alpha_decay=0.98,cycle=args.cycle)
    cs = FireflyAlgorithm(fa_config,problem)
    print(cs.run())
    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Algorithm':<30} {'Best Fitness':>20}")
    print("-" * 55)
    print(f"{'Genetic Algorithm':<30} {ga.best_fitness:>20.10f}")
    print(f"{'Differential Evolution':<30} {de.best_fitness:>20.10f}")
    for es in es_results:
        print(f"{es.name:<30} {es.best_fitness:>20.10f}")
    print(f"{'Artificial Bee Colony':<30} {abc.best_fitness:>20.10f}")
    print(f"{'Cuckoo Search':<30} {cs.best_fitness:>20.10f}")
    print("=" * 55)

    # ── Convergence history ──────────────────────────────────────────
    # GA / DE history stores best-solution vectors; extract fitness
    ga_fitness_history = [float(problem.eval(sol)) for sol in ga.history]
    de_fitness_history = [float(problem.eval(sol)) for sol in de.history]
    # ES history already stores scalar best-fitness per generation
    es_fitness_histories = [es.history for es in es_results]
    abc_fitness_history = [float(problem.eval(sol)) for sol in abc.history]
    cs_fitness_history = [float(problem.eval(sol)) for sol in cs.history]

    if not HAS_PLT:
        print("\nmatplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: convergence curves
    axes[0].plot(ga_fitness_history, label="GA", linewidth=1.5)
    axes[0].plot(de_fitness_history, label="DE", linewidth=1.5)
    axes[0].plot(abc_fitness_history, label="ABC", linewidth=1.5)
    axes[0].plot(cs_fitness_history, label="CS", linewidth=1.5)
    for es, hist in zip(es_results, es_fitness_histories):
        axes[0].plot(hist, label=es.name, linewidth=1.5)
    axes[0].set_title(f"Convergence on {problem._name} (dim={args.dim})")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Fitness")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, which="both", linestyle="--")
    axes[0].legend(fontsize=8)

    # Subplot 2: final best solutions (first 2 dims if available)
    if args.dim >= 2:
        ax2 = axes[1]
        markers = ["^", "s", "o", "D", "P", "X", "*", "8"]
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:cyan",
            "tab:pink",
        ]
        all_algorithms = [
            ("GA", ga),
            ("DE", de),
            *[(es.name, es) for es in es_results],
            ("ABC", abc),
            ("CS", cs)
        ]
        for i, (label, algo) in enumerate(all_algorithms):
            sol = algo.best_solution
            ax2.scatter(
                sol[0],
                sol[1],
                s=120,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=f"{label} ({algo.best_fitness:.4f})",
                zorder=5,
            )
        ax2.scatter(
            0, 0, s=80, marker="*", color="black", label="Global Optimum", zorder=6
        )
        ax2.set_title("Best Solutions (first 2 dims)")
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
