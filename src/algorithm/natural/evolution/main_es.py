"""main_es.py — Test all Evolution Strategy variants on benchmark problems.

Supports all five benchmark problems (sphere, rastrigin, ackley, cigar, ridge)
and runs all four ES variants defined in algorithm/natural/evolution/es/.

Usage (from src/)
-----------------
    python -m algorithm.natural.evolution.main_es
    python -m algorithm.natural.evolution.main_es --problem cigar --dim 20
    python -m algorithm.natural.evolution.main_es --problem ridge --cycle 1000 --seed 0
"""

import random
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from problems import Sphere, Rastrigin, Ackley, Cigar, Ridge
from algorithm.natural.evolution.es.es import EvolutionStrategy, ESVariant
from algorithm.natural.evolution.es import (
    OnePlusOneESParameter,
    SelfAdaptiveESParameter,
    CMAESParameter,
    MuRhoPlusLambdaESParameter,
)


def set_seed(seed: int | float | None):
    """Set random seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")


def run_es(problem, n_dim: int, cycle: int, variant: ESVariant) -> EvolutionStrategy:
    """Configure sensible defaults for *variant* and run it on *problem*."""
    match variant:
        case ESVariant.ONE_PLUS_ONE:
            config = OnePlusOneESParameter(sigma=1.0, cycle=cycle)
            desc = "σ=1.0"
        case ESVariant.SELF_ADAPTIVE:
            lam = max(20, 4 * n_dim)
            mu = lam // 4
            config = SelfAdaptiveESParameter(
                mu=mu, lam=lam, rho=2, sigma_init=1.0, cycle=cycle
            )
            desc = f"μ={mu}, λ={lam}, ρ=2"
        case ESVariant.CMA_ES:
            config = CMAESParameter(sigma=1.0, cycle=cycle)
            desc = "σ=1.0 (auto λ/μ)"
        case ESVariant.MU_RHO_PLUS_LAMBDA:
            lam = max(20, 4 * n_dim)
            mu = lam // 4
            config = MuRhoPlusLambdaESParameter(
                mu=mu, rho=2, lam=lam, sigma_init=1.0, cycle=cycle
            )
            desc = f"μ={mu}, ρ=2, λ={lam}"

    es = EvolutionStrategy(variant=variant, configuration=config, problem=problem)

    print(f"\nRunning {variant.value} [{desc}]...")
    best = es.run()

    print(f"  Best Fitness : {es.best_fitness:.10f}")
    print(f"  Best Solution: {best[:min(5, n_dim)]}{'...' if n_dim > 5 else ''}")

    return es


def main():
    parser = argparse.ArgumentParser(
        description="Test all ES variants on continuous benchmark problems"
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
        choices=["sphere", "rastrigin", "ackley", "cigar", "ridge"],
        help="Benchmark problem (default: ackley)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    all_problems = {
        "sphere":    Sphere,
        "rastrigin": Rastrigin,
        "ackley":    Ackley,
        "cigar":     Cigar,
        "ridge":     Ridge,
    }
    problem = all_problems[args.problem](n_dim=args.dim)
    print(f"Problem : {problem._name}, Dimensions: {problem._n_dim}")
    print(f"Bounds  : {problem._bounds[0]}")

    # ── Run all four ES variants ─────────────────────────────────────
    es_1p1  = run_es(problem, args.dim, args.cycle, ESVariant.ONE_PLUS_ONE)
    es_sa   = run_es(problem, args.dim, args.cycle, ESVariant.SELF_ADAPTIVE)
    es_cma  = run_es(problem, args.dim, args.cycle, ESVariant.CMA_ES)
    es_plus = run_es(problem, args.dim, args.cycle, ESVariant.MU_RHO_PLUS_LAMBDA)

    all_es = [es_1p1, es_sa, es_cma, es_plus]
    labels = ["(1+1)-ES", "Self-Adaptive (μ,λ)-ES", "CMA-ES", "(μ/ρ+λ)-ES"]

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Algorithm':<30} {'Best Fitness':>20}")
    print("-" * 55)
    for label, es in zip(labels, all_es):
        print(f"{label:<30} {es.best_fitness:>20.10f}")
    print("=" * 55)

    if not HAS_PLT:
        print("\nmatplotlib not installed — skipping plots.")
        return

    # ── Plots ────────────────────────────────────────────────────────
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Evolution Strategies on {problem._name} (dim={args.dim}, cycles={args.cycle})",
        fontsize=13,
    )

    # Subplot 1: convergence curves
    # All ES variants store scalar fitness values in history directly.
    ax = axes[0]
    for es, label, color in zip(all_es, labels, colors):
        ax.plot(es.history, label=label, linewidth=1.5, color=color)

    ax.set_title("Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    all_hist = [v for es in all_es for v in es.history]
    if min(all_hist) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.legend(fontsize=9)

    # Subplot 2: bar chart of final best fitness per variant
    ax2 = axes[1]
    values = [es.best_fitness for es in all_es]
    short_labels = ["(1+1)-ES", "SA-(μ,λ)", "CMA-ES", "(μ/ρ+λ)-ES"]
    bars = ax2.bar(short_labels, values, color=colors, alpha=0.8)

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4g}",
            ha="center", va="bottom", fontsize=8,
        )

    ax2.set_title("Final Best Fitness")
    ax2.set_ylabel("Best Fitness")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
