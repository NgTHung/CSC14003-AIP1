"""Demo script to test local search algorithms (Hill Climbing, Steepest-Ascent HC)."""

import sys
sys.path.append('src')

import matplotlib.pyplot as plt

from AIP.problems.discrete.nqueens import NQueensProblem
from AIP.algorithm.local.HillClimbing import HillClimbing
from AIP.algorithm.local.SteepestAscentHC import SteepestAscentHillClimbing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_result(name: str, state, value: float, stats: dict, problem=None):
    """Print the result of a local search algorithm run."""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Final value   : {value}")
    print(f"  Iterations    : {stats['iterations']}")
    print(f"  Initial value : {stats['initial_value']}")
    print(f"  Improvement   : {stats['improvement']:.4f}")
    if problem is not None and hasattr(problem, 'is_solution'):
        solved = problem.is_solution(state)
        print(f"  Solved        : {'Yes' if solved else 'No'}")


def plot_value_history(all_histories: dict[str, list], title: str, filename: str):
    """
    Plot objective value over iterations for each algorithm.

    Parameters
    ----------
    all_histories : dict[str, list]
        Mapping algorithm_name -> history (list of dicts with 'value').
    title : str
        Plot title.
    filename : str
        Output image filename.
    """
    plt.figure(figsize=(10, 5))
    for name, history in all_histories.items():
        iters = [h['iteration'] for h in history]
        values = [h['value'] for h in history]
        plt.plot(iters, values, marker='o', markersize=4, label=name)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (conflicts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_comparison_bar(results: list[dict], title: str, filename: str):
    """
    Bar chart comparing final value and iterations.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys 'name', 'final_value', 'iterations'.
    title : str
        Plot super-title.
    filename : str
        Output image filename.
    """
    names = [r['name'] for r in results]
    final_vals = [r['final_value'] for r in results]
    iterations = [r['iterations'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Final value
    bars1 = axes[0].bar(names, final_vals, color='steelblue')
    axes[0].set_title('Final Value (lower is better)')
    axes[0].set_ylabel('Conflicts')
    for bar, val in zip(bars1, final_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    # Iterations
    bars2 = axes[1].bar(names, iterations, color='coral')
    axes[1].set_title('Iterations')
    axes[1].set_ylabel('Count')
    for bar, val in zip(bars2, iterations):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Test N-Queens
# ---------------------------------------------------------------------------

def test_nqueens(n: int = 8, num_trials: int = 5):
    """
    Test hill-climbing variants on the N-Queens problem.

    Parameters
    ----------
    n : int
        Board size.
    num_trials : int
        Number of random restarts to average over.
    """
    print("\n" + "=" * 60)
    print(f"  N-QUEENS (N={n})  —  Minimize attacking pairs")
    print("=" * 60)

    problem = NQueensProblem(n=n)

    # --- Configuration ---
    hc_config = {'max_iterations': 1000}
    sahc_max_iter = 1000

    # Single run with detailed output
    print("\n>>> Single Run <<<")

    hc = HillClimbing(hc_config, problem)
    hc_state, hc_val = hc.run()
    hc_stats = hc.get_statistics()
    print_result("Hill Climbing (First-Choice)", hc_state, hc_val, hc_stats, problem)
    if problem.is_solution(hc_state):
        print("  Board:")
        print("  " + problem.display(hc_state).replace('\n', '\n  '))

    sahc = SteepestAscentHillClimbing(problem, max_iterations=sahc_max_iter)
    sahc_state, sahc_val = sahc.run()
    sahc_stats = sahc.get_statistics()
    print_result("Steepest-Ascent HC", sahc_state, sahc_val, sahc_stats, problem)
    if problem.is_solution(sahc_state):
        print("  Board:")
        print("  " + problem.display(sahc_state).replace('\n', '\n  '))

    # Plot single-run value history
    plot_value_history(
        {
            'Hill Climbing': hc.history,
            'Steepest-Ascent HC': sahc.history,
        },
        f'{n}-Queens — Value over Iterations (single run)',
        'local_search_nqueens_single.png',
    )

    # --- Multiple trials ---
    print(f"\n>>> {num_trials} Random Restarts <<<")

    hc_successes = 0
    sahc_successes = 0
    hc_total_iters = 0
    sahc_total_iters = 0

    for trial in range(num_trials):
        hc2 = HillClimbing(hc_config, problem)
        s, v = hc2.run()
        hc_total_iters += len(hc2.history)
        if problem.is_solution(s):
            hc_successes += 1

        sahc2 = SteepestAscentHillClimbing(problem, max_iterations=sahc_max_iter)
        s, v = sahc2.run()
        sahc_total_iters += len(sahc2.history)
        if problem.is_solution(s):
            sahc_successes += 1

    print(f"  Hill Climbing      : {hc_successes}/{num_trials} solved "
          f"(avg iters: {hc_total_iters / num_trials:.1f})")
    print(f"  Steepest-Ascent HC : {sahc_successes}/{num_trials} solved "
          f"(avg iters: {sahc_total_iters / num_trials:.1f})")

    # Bar comparison from single run
    plot_comparison_bar(
        [
            {'name': 'Hill Climbing', 'final_value': hc_val, 'iterations': hc_stats['iterations']},
            {'name': 'Steepest-Ascent', 'final_value': sahc_val, 'iterations': sahc_stats['iterations']},
        ],
        f'{n}-Queens — Algorithm Comparison',
        'local_search_nqueens_comparison.png',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_nqueens(n=8, num_trials=10)
