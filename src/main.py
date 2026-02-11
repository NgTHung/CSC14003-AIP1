import random
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from problems import Sphere, Rastrigin, Ackley
from algorithm.natural.evolution.ga import (
    GeneticAlgorithm,
    GAParameter,
    SelectionMethod,
    CrossoverMethod,
)
from algorithm.natural.evolution.de import (
    DifferentialEvolution,
    DEParameter,
    MutationStrategy,
    DECrossoverType,
)


def set_seed(seed: int | float | None):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
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
    print(f"  pop_size={config.pop_size}, n_bits={config.n_bits}, "
          f"pc={config.pc}, pm={config.pm}, "
          f"selection={config.selection.value}, "
          f"crossover={config.crossover.value}")

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

    print(f"\nRunning {de.name} (DE/{config.strategy.value}/{config.crossover_type.value})...")
    print(f"  pop_size={config.pop_size}, F={config.F}, Cr={config.Cr}")

    best = de.run()

    print(f"  Best Fitness : {de.best_fitness:.10f}")
    print(f"  Best Solution: {best[:min(5, n_dim)]}{'...' if n_dim > 5 else ''}")

    return de


def main():
    parser = argparse.ArgumentParser(
        description="Test GA and DE on benchmark problems"
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
        help="Number of generations (default: 500)",
    )
    parser.add_argument(
        "--problem", type=str, default="ackley",
        choices=["sphere", "rastrigin", "ackley"],
        help="Benchmark problem (default: ackley)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Select problem
    problems = {
        "sphere": Sphere,
        "rastrigin": Rastrigin,
        "ackley": Ackley,
    }
    problem = problems[args.problem](n_dim=args.dim)
    print(f"Problem: {problem._name}, Dimensions: {problem._n_dim}")
    print(f"Bounds : {problem._bounds[0]}")

    # ── Run both algorithms ──────────────────────────────────────────
    ga = run_ga(problem, args.dim, args.cycle)
    de = run_de(problem, args.dim, args.cycle)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Algorithm':<25} {'Best Fitness':>20}")
    print("-" * 50)
    print(f"{'Genetic Algorithm':<25} {ga.best_fitness:>20.10f}")
    print(f"{'Differential Evolution':<25} {de.best_fitness:>20.10f}")
    print("=" * 50)

    # ── Convergence history ──────────────────────────────────────────
    # GA history stores best-solution vectors; extract fitness per gen
    ga_fitness_history = [
        float(problem.eval(sol)) for sol in ga.history
    ]
    # DE history also stores best-solution vectors
    de_fitness_history = [
        float(problem.eval(sol)) for sol in de.history
    ]

    if not HAS_PLT:
        print("\nmatplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: convergence curves together
    axes[0].plot(ga_fitness_history, label="GA", linewidth=1.5)
    axes[0].plot(de_fitness_history, label="DE", linewidth=1.5)
    axes[0].set_title(f"Convergence on {problem._name} (dim={args.dim})")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best Fitness")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3, which="both", linestyle="--")
    axes[0].legend()

    # Subplot 2: final best solutions (first 2 dims if available)
    if args.dim >= 2:
        ax2 = axes[1]
        ga_sol = ga.best_solution
        de_sol = de.best_solution
        ax2.scatter(ga_sol[0], ga_sol[1], s=120, marker="^",
                    color="tab:blue", label=f"GA ({ga.best_fitness:.4f})", zorder=5)
        ax2.scatter(de_sol[0], de_sol[1], s=120, marker="s",
                    color="tab:orange", label=f"DE ({de.best_fitness:.4f})", zorder=5)
        ax2.scatter(0, 0, s=80, marker="*", color="red",
                    label="Global Optimum", zorder=6)
        ax2.set_title("Best Solutions (first 2 dims)")
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
