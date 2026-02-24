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

import matplotlib.pyplot as plt
import numpy as np

from problems.continuous import Rastrigin, Ackley, Sphere
from problems.discrete.tsp import TSP
from algorithm.natural.human.tlbo import TLBO, TLBOConfig
from algorithm.natural.human.sfo import SFO, SFOConfig


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int | float | None) -> None:
    """Fix the global random seed for reproducibility.

    Parameters
    ----------
    seed : int or float or None
        Seed value.  Pass ``None`` to leave the RNG in its default state.
    """
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
        print(f"Random seed set to: {seed}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, run SFO on Rastrigin, and plot convergence."""

    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Run optimization algorithms on benchmark problems"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # -----------------------------------------------------------------------
    # 1.  Problem setup – Problem in 10 dimensions
    # -----------------------------------------------------------------------
    # problem = Rastrigin(n_dim=10) # Change problem name and dimensions here
    problem = Ackley(n_dim=10)
    # problem = Sphere(n_dim=10)

    print(f"Problem    : {problem._name}")
    print(f"Dimensions : {problem._n_dim}")
    print(f"Bounds     : {problem._bounds[0]}  (same for all dims)")

    # -----------------------------------------------------------------------
    # 2.  Algorithm configuration
    # -----------------------------------------------------------------------
    config = TLBOConfig(pop_size=100, iterations=5000, minimization=True) # TLBO configuration example
    # config = SFOConfig(
    #     pop_size   = 100,
    #     iterations = 300,
    #     minimization = True,
    #     w          = 0.9,    # initial inertia weight
    #     w_decay    = 0.99,   # multiplicative decay per iteration
    #     c_attract  = 1.5,    # attraction towards global best
    #     c_social   = 1.5,    # attraction towards crowd mean
    # )
    

    # -----------------------------------------------------------------------
    # 3.  Instantiate and run the optimiser
    # -----------------------------------------------------------------------
    optimizer = TLBO(configuration=config, problem=problem) # TLBO instance
    # optimizer = SFO(configuration=config, problem=problem)
    print(f"\nRunning {optimizer.name} ...")
    best_solution = optimizer.run()

    # -----------------------------------------------------------------------
    # 4.  Report results
    # -----------------------------------------------------------------------
    print("\n=== RESULTS ===")
    print(f"Best Fitness           : {optimizer.best_fitness}")
    print(f"Best Solution (first 5): {best_solution[:5]}")
    print(f"Best Solution (full)   : {best_solution}")

    # -----------------------------------------------------------------------
    # 5.  Convergence plot
    # -----------------------------------------------------------------------
    iterations = list(range(1, len(optimizer.history) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        iterations,
        optimizer.history,
        color="royalblue",
        linewidth=2,
        label=f"{optimizer.name}  (pop={config.pop_size}, iter={config.iterations})",
    )

    # Mark the final best value with a star marker
    ax.scatter(
        iterations[-1],
        optimizer.history[-1],
        color="crimson",
        zorder=5,
        s=80,
        label=f"Final best ≈ {optimizer.best_fitness:.4e}",
    )

    ax.set_yscale("log")                          # log scale highlights early gains
    ax.set_title(
        f"{optimizer.name} Convergence on {problem._name} Function  (n_dim={problem._n_dim})",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
