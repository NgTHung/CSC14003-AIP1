"""Demo script to test classical graph search algorithms (DFS, BFS, UCS, Greedy, A*)."""

import sys
sys.path.append('src')

import matplotlib.pyplot as plt

from AIP.problems.discrete.simple_graph import SimpleGraph
from AIP.problems.discrete.romania_map import RomaniaMap
from AIP.algorithm.classical.DFS import DepthFirstSearch
from AIP.algorithm.classical.BFS import BreadthFirstSearch
from AIP.algorithm.classical.UCS import UniformCostSearch
from AIP.algorithm.classical.GreedyBestFirst import GreedyBestFirstSearch
from AIP.algorithm.classical.AStar import AStarSearch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_result(algo, label: str):
    """Print the result of a graph search algorithm run."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    if algo.best_solution is not None:
        print(f"  Path found   : {' -> '.join(str(s) for s in algo.best_solution)}")
        print(f"  Path cost    : {algo.best_fitness}")
        print(f"  Nodes explored: {len(algo.history)}")
    else:
        print("  No path found.")
        print(f"  Nodes explored: {len(algo.history)}")


def plot_comparison(results: list[dict], title: str):
    """
    Bar‑chart comparing algorithms on nodes explored and path cost.

    Parameters
    ----------
    results : list[dict]
        Each dict has keys 'name', 'explored', 'cost'.
    title : str
        Plot super‑title.
    """
    names = [r['name'] for r in results]
    explored = [r['explored'] for r in results]
    costs = [r['cost'] if r['cost'] is not None else 0 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Nodes explored
    bars1 = axes[0].bar(names, explored, color='steelblue')
    axes[0].set_title('Nodes Explored')
    axes[0].set_ylabel('Count')
    for bar, val in zip(bars1, explored):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(val), ha='center', va='bottom', fontsize=9)

    # Path cost
    bars2 = axes[1].bar(names, costs, color='coral')
    axes[1].set_title('Path Cost')
    axes[1].set_ylabel('Cost')
    for bar, val in zip(bars2, costs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}" if val else "N/A", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('graph_search_comparison.png', dpi=150)
    plt.show()


def plot_explored_over_steps(all_histories: dict[str, list], title: str):
    """
    Line plot of explored‑node count at each step for every algorithm.

    Parameters
    ----------
    all_histories : dict[str, list]
        Mapping algorithm_name -> history (list of dicts).
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 5))
    for name, history in all_histories.items():
        steps = list(range(len(history)))
        explored_counts = [h['explored_count'] for h in history]
        plt.plot(steps, explored_counts, marker='o', markersize=3, label=name)

    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Explored Nodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_search_explored.png', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Test on Romania Map
# ---------------------------------------------------------------------------

def test_romania_map():
    print("\n" + "=" * 60)
    print("  ROMANIA MAP  —  Arad ➜ Bucharest")
    print("=" * 60)

    problem = RomaniaMap(initial_state='Arad', goal_state='Bucharest')
    config: dict = {}

    algorithms = [
        ("DFS",          DepthFirstSearch(config, problem)),
        ("BFS",          BreadthFirstSearch(config, problem)),
        ("UCS",          UniformCostSearch(config, problem)),
        ("Greedy BFS",   GreedyBestFirstSearch(config, problem)),
        ("A*",           AStarSearch(config, problem)),
    ]

    results = []
    histories = {}

    for name, algo in algorithms:
        algo.run()
        print_result(algo, name)
        results.append({
            'name': name,
            'explored': len(algo.history),
            'cost': algo.best_fitness,
        })
        histories[name] = algo.history

    # Plots
    plot_comparison(results, "Romania Map — Algorithm Comparison")
    plot_explored_over_steps(histories, "Romania Map — Explored Nodes per Step")


# ---------------------------------------------------------------------------
# Test on Simple Graph
# ---------------------------------------------------------------------------

def test_simple_graph():
    print("\n" + "=" * 60)
    print("  SIMPLE GRAPH  —  A ➜ D")
    print("=" * 60)

    problem = SimpleGraph.create_simple_graph()
    config: dict = {}

    algorithms = [
        ("DFS",          DepthFirstSearch(config, problem)),
        ("BFS",          BreadthFirstSearch(config, problem)),
        ("UCS",          UniformCostSearch(config, problem)),
        ("Greedy BFS",   GreedyBestFirstSearch(config, problem)),
        ("A*",           AStarSearch(config, problem)),
    ]

    results = []
    histories = {}

    for name, algo in algorithms:
        algo.run()
        print_result(algo, name)
        results.append({
            'name': name,
            'explored': len(algo.history),
            'cost': algo.best_fitness,
        })
        histories[name] = algo.history

    plot_comparison(results, "Simple Graph — Algorithm Comparison")
    plot_explored_over_steps(histories, "Simple Graph — Explored Nodes per Step")


# ---------------------------------------------------------------------------
# Test on Grid Graph
# ---------------------------------------------------------------------------

def test_grid_graph():
    print("\n" + "=" * 60)
    print("  GRID GRAPH 5x5  —  (0,0) ➜ (4,4)")
    print("=" * 60)

    problem = SimpleGraph.create_grid_graph(rows=5, cols=5)
    config: dict = {}

    algorithms = [
        ("DFS",          DepthFirstSearch(config, problem)),
        ("BFS",          BreadthFirstSearch(config, problem)),
        ("UCS",          UniformCostSearch(config, problem)),
    ]

    results = []
    histories = {}

    for name, algo in algorithms:
        algo.run()
        print_result(algo, name)
        results.append({
            'name': name,
            'explored': len(algo.history),
            'cost': algo.best_fitness,
        })
        histories[name] = algo.history

    plot_comparison(results, "Grid 5x5 — Algorithm Comparison")
    plot_explored_over_steps(histories, "Grid 5x5 — Explored Nodes per Step")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_romania_map()
    test_simple_graph()
    test_grid_graph()
