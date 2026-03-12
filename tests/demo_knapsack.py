"""Demo script for the 0/1 Knapsack Problem with multiple algorithm families.

Demonstrates:
1. Classical graph search (DFS, BFS, UCS, Greedy, A*)
2. Local search (Hill Climbing)
3. Physics-inspired (Simulated Annealing, Harmony Search)
4. Biology-inspired (Artificial Bee Colony, Cuckoo Search, Firefly Algorithm)
"""

import sys
sys.path.append("src")

import time
import numpy as np
from AIP.problems.discrete.knapsack import Knapsack
from AIP.algorithm.classical.DFS import DepthFirstSearch
from AIP.algorithm.classical.BFS import BreadthFirstSearch
from AIP.algorithm.classical.UCS import UniformCostSearch
from AIP.algorithm.classical.GreedyBestFirst import GreedyBestFirstSearch
from AIP.algorithm.classical.AStar import AStarSearch
from AIP.algorithm.local.HillClimbing import HillClimbing, HillClimbingParameter
from AIP.algorithm.natural.physic.SA import SimulatedAnnealing, SimulatedAnnealingParameter
from AIP.algorithm.natural.physic.HS import HarmonySearch
from AIP.algorithm.natural.biology.abc import ArtificialBeeColony, ABCParameter
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter


# ============================================================================
# Helper — pretty-print a decoded knapsack solution
# ============================================================================

def _print_knapsack_result(info: dict, problem: Knapsack, indent: str = "  "):
    """Print a decoded knapsack solution (selected_items, total_value, total_weight)."""
    item_names = [f"item{i}" for i in info["selected_items"]]
    print(f"{indent}Selected items : {info['selected_items']}  {item_names}")
    print(f"{indent}Total value    : {info['total_value']:.0f}")
    print(f"{indent}Total weight   : {info['total_weight']:.0f}  (capacity {problem.capacity:.0f})")
    feasible = info["total_weight"] <= problem.capacity
    print(f"{indent}Feasible?      : {feasible}")


# ============================================================================
# Knapsack — Graph Search
# ============================================================================

def demo_knapsack_graph_search():
    """Run classical graph search algorithms on a small Knapsack instance."""
    problem = Knapsack.create_small()

    print("=" * 65)
    print("  Knapsack — Classical Graph Search (small instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Weights : {problem.weights.tolist()}")
    print(f"  Values  : {problem.values.tolist()}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config: dict = {}
    algorithms = [
        ("DFS",        DepthFirstSearch(config, problem)),
        ("BFS",        BreadthFirstSearch(config, problem)),
        ("UCS",        UniformCostSearch(config, problem)),
        ("Greedy BFS", GreedyBestFirstSearch(config, problem)),
        ("A*",         AStarSearch(config, problem)),
    ]

    for name, algo in algorithms:
        t0 = time.perf_counter()
        algo.run()
        elapsed = time.perf_counter() - t0
        print(f"\n{'─' * 55}")
        print(f"  {name}")
        print(f"{'─' * 55}")
        if algo.best_solution is not None:
            info = problem.decode_path(algo.best_solution)
            _print_knapsack_result(info, problem)
            print(f"  Nodes explored: {len(algo.history)}")
        else:
            print("  No solution found.")
            print(f"  Nodes explored: {len(algo.history)}")
        print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Local Search (Hill Climbing)
# ============================================================================

def demo_knapsack_local_search():
    """Run Hill Climbing on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Hill Climbing (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Weights : {problem.weights.tolist()}")
    print(f"  Values  : {problem.values.tolist()}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = HillClimbingParameter(iteration=500)
    algo = HillClimbing(config, problem)
    initial = problem.sample(1).flatten()
    t0 = time.perf_counter()
    algo.run(initial_state=initial)
    elapsed = time.perf_counter() - t0

    if algo.best_solution is not None:
        best = np.asarray(algo.best_solution).flatten()
        info = problem.decode_binary(best)
        _print_knapsack_result(info, problem)
    else:
        print("  No solution found.")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Simulated Annealing
# ============================================================================

def demo_knapsack_sa():
    """Run Simulated Annealing on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Simulated Annealing (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = SimulatedAnnealingParameter(
        initial_temperature=500.0,
        cooling_rate=0.995,
        min_temperature=0.01,
        max_iterations=5000,
        n_flips=1,
    )
    algo = SimulatedAnnealing(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_binary(best)
    _print_knapsack_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Harmony Search
# ============================================================================

def demo_knapsack_hs():
    """Run Harmony Search on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Harmony Search (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = {
        "hms": 30,
        "hmcr": 0.9,
        "par": 0.3,
        "bw": 0.1,
        "max_iterations": 3000,
    }
    algo = HarmonySearch(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_binary(best)
    _print_knapsack_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Artificial Bee Colony
# ============================================================================

def demo_knapsack_abc():
    """Run Artificial Bee Colony on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Artificial Bee Colony (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = ABCParameter(n_bees=30, limit=50, iteration=500)
    algo = ArtificialBeeColony(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_binary(best)
    _print_knapsack_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Cuckoo Search
# ============================================================================

def demo_knapsack_cs():
    """Run Cuckoo Search on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Cuckoo Search (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = CuckooSearchParameter(n_nests=25, pa=0.25, alpha=0.01, beta=1.5, iteration=500)
    algo = CuckooSearch(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_binary(best)
    _print_knapsack_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Knapsack — Firefly Algorithm
# ============================================================================

def demo_knapsack_fa():
    """Run Firefly Algorithm on a Knapsack instance."""
    problem = Knapsack.create_medium()

    print("\n" + "=" * 65)
    print("  Knapsack — Firefly Algorithm (medium instance)")
    print(f"  Items   : {problem.n_items}")
    print(f"  Capacity: {problem.capacity}")
    print("=" * 65)

    config = FireflyParameter(
        n_fireflies=25, alpha=0.5, beta0=1.0,
        gamma=1.0, alpha_decay=0.97, cycle=300,
    )
    algo = FireflyAlgorithm(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_binary(best)
    _print_knapsack_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("#  0/1 KNAPSACK PROBLEM")
    print("#" * 65)

    demo_knapsack_graph_search()
    demo_knapsack_local_search()
    demo_knapsack_sa()
    demo_knapsack_hs()
    demo_knapsack_abc()
    demo_knapsack_cs()
    demo_knapsack_fa()
