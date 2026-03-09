"""Demo script to test physics-inspired optimization algorithms (SA, GSA, HS)."""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt

from AIP.problems.continuous.sphere import Sphere
from AIP.problems.continuous.rastrigin import Rastrigin
from AIP.problems.continuous.ackley import Ackley
from AIP.algorithm.physics.SA import SimulatedAnnealing
from AIP.algorithm.physics.GSA import GravitationalSearchAlgorithm
from AIP.algorithm.physics.HS import HarmonySearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_result(algo, problem_name: str):
    """Print the result of a physics-inspired algorithm run."""
    print(f"\n{'─' * 50}")
    print(f"  {algo.name}")
    print(f"{'─' * 50}")
    print(f"  Problem        : {problem_name}")
    print(f"  Best fitness   : {algo.best_fitness:.6e}")
    print(f"  Best solution  : {np.array2string(algo.best_solution, precision=4, suppress_small=True)}")
    print(f"  Iterations     : {len(algo.history)}")


def plot_convergence(all_histories: dict[str, list], title: str, filename: str):
    """
    Plot best-fitness convergence over iterations for every algorithm.

    Parameters
    ----------
    all_histories : dict[str, list]
        Mapping algorithm_name -> history list of dicts with 'best_fitness'.
    title : str
        Plot title.
    filename : str
        Output image filename.
    """
    plt.figure(figsize=(10, 5))
    for name, history in all_histories.items():
        iters = [h['iteration'] for h in history]
        best = [h['best_fitness'] for h in history]
        plt.plot(iters, best, label=name)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_sa_temperature(history: list, title: str, filename: str):
    """
    Dual-axis plot: temperature decay and energy / best energy for SA.

    Parameters
    ----------
    history : list
        SA history (list of dicts with 'iteration', 'temperature', 'fitness', 'best_fitness').
    title : str
        Plot title.
    filename : str
        Output image filename.
    """
    iters = [h['iteration'] for h in history]
    temps = [h['temperature'] for h in history]
    energies = [h['fitness'] for h in history]
    best_energies = [h['best_fitness'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_temp = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Temperature', color=color_temp)
    ax1.plot(iters, temps, color=color_temp, alpha=0.6, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color_temp)

    ax2 = ax1.twinx()
    color_energy = 'tab:blue'
    ax2.set_ylabel('Energy', color=color_energy)
    ax2.plot(iters, energies, color='lightblue', alpha=0.4, linewidth=0.5, label='Current energy')
    ax2.plot(iters, best_energies, color=color_energy, label='Best energy')
    ax2.tick_params(axis='y', labelcolor=color_energy)

    fig.suptitle(title)
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.88))
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_comparison_bar(results: list[dict], title: str, filename: str):
    """
    Bar chart comparing best fitness and iterations across algorithms.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys 'name', 'best_fitness', 'iterations'.
    title : str
        Plot super-title.
    filename : str
        Output image filename.
    """
    names = [r['name'] for r in results]
    fitness = [r['best_fitness'] for r in results]
    iterations = [r['iterations'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    bars1 = axes[0].bar(names, fitness, color='steelblue')
    axes[0].set_title('Best Fitness (lower is better)')
    axes[0].set_ylabel('Fitness')
    for bar, val in zip(bars1, fitness):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2e}", ha='center', va='bottom', fontsize=8)

    bars2 = axes[1].bar(names, iterations, color='coral')
    axes[1].set_title('Iterations')
    axes[1].set_ylabel('Count')
    for bar, val in zip(bars2, iterations):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def run_all_on_problem(problem, problem_name: str, tag: str):
    """
    Run SA, GSA, HS on a single continuous problem.

    Prints results and generates convergence + bar-chart plots.
    """
    print("\n" + "=" * 60)
    print(f"  {problem_name}  —  Global optimum: f(0,...,0) = 0")
    print("=" * 60)

    # --- Configurations ---
    sa_config = {
        'initial_temperature': 100.0,
        'cooling_rate': 0.95,
        'min_temperature': 0.01,
        'max_iterations': 1000,
        'step_size': 0.5,
    }
    gsa_config = {
        'pop_size': 30,
        'max_iterations': 200,
        'G0': 100.0,
        'alpha': 20.0,
    }
    hs_config = {
        'hms': 20,
        'hmcr': 0.9,
        'par': 0.3,
        'bw': 0.5,
        'max_iterations': 1000,
    }

    # --- Run ---
    sa = SimulatedAnnealing(sa_config, problem)
    sa.run()
    print_result(sa, problem_name)

    gsa = GravitationalSearchAlgorithm(gsa_config, problem)
    gsa.run()
    print_result(gsa, problem_name)

    hs = HarmonySearch(hs_config, problem)
    hs.run()
    print_result(hs, problem_name)

    # --- Plots ---
    # Convergence
    plot_convergence(
        {'SA': sa.history, 'GSA': gsa.history, 'HS': hs.history},
        f'{problem_name} — Convergence',
        f'physics_{tag}_convergence.png',
    )

    # SA temperature + energy
    plot_sa_temperature(
        sa.history,
        f'{problem_name} — SA Temperature & Energy',
        f'physics_{tag}_sa_temp.png',
    )

    # Bar comparison
    plot_comparison_bar(
        [
            {'name': 'SA',  'best_fitness': sa.best_fitness, 'iterations': len(sa.history)},
            {'name': 'GSA', 'best_fitness': gsa.best_fitness, 'iterations': len(gsa.history)},
            {'name': 'HS',  'best_fitness': hs.best_fitness, 'iterations': len(hs.history)},
        ],
        f'{problem_name} — Algorithm Comparison',
        f'physics_{tag}_comparison.png',
    )


def test_sphere():
    problem = Sphere(n_dim=2)
    run_all_on_problem(problem, 'Sphere 2D', 'sphere')


def test_rastrigin():
    problem = Rastrigin(n_dim=2)
    run_all_on_problem(problem, 'Rastrigin 2D', 'rastrigin')


def test_ackley():
    problem = Ackley(n_dim=2)
    run_all_on_problem(problem, 'Ackley 2D', 'ackley')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_sphere()
    test_rastrigin()
    test_ackley()
