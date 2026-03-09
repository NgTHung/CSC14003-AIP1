"""Demo script for the Travelling Salesman Problem (TSP) with multiple algorithm families.

Demonstrates:
1. Classical graph search (DFS, BFS, UCS, Greedy, A*)
2. Local search (Hill Climbing)
3. Physics-inspired (Simulated Annealing, Harmony Search)
4. Biology-inspired (Artificial Bee Colony, Cuckoo Search, Firefly Algorithm)
5. Ant Colony Optimization (Ant System, Ant Colony System, MAX-MIN Ant System)
"""

import sys
sys.path.append("src")

import time
import numpy as np
from AIP.problems.discrete.tsp import TSP
from AIP.algorithm.classical.DFS import DepthFirstSearch
from AIP.algorithm.classical.BFS import BreadthFirstSearch
from AIP.algorithm.classical.UCS import UniformCostSearch
from AIP.algorithm.classical.GreedyBestFirst import GreedyBestFirstSearch
from AIP.algorithm.classical.AStar import AStarSearch
from AIP.algorithm.local.HillClimbing import HillClimbing, HillClimbingParameter
from AIP.algorithm.natural.physic.SA import SimulatedAnnealing
from AIP.algorithm.natural.physic.HS import HarmonySearch
from AIP.algorithm.natural.biology.abc import ArtificialBeeColony, ABCParameter
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter
from AIP.algorithm.natural.biology.aco import AntSystem, AntSystemParameter
from AIP.algorithm.natural.biology.aco import ACS, ACSParameter
from AIP.algorithm.natural.biology.aco import MMAS, MMASParameter


# ============================================================================
# Helper — pretty-print a decoded TSP solution
# ============================================================================

def _print_tsp_result(info: dict, indent: str = "  "):
    """Print a decoded TSP solution (tour, distance)."""
    print(f"{indent}Tour           : {info['tour']}")
    print(f"{indent}Tour (names)   : {' -> '.join(info['tour_names'])}")
    print(f"{indent}Total distance : {info['total_distance']:.2f}")


# ============================================================================
# TSP — Graph Search
# ============================================================================

def demo_tsp_graph_search():
    """Run classical graph search algorithms on a small TSP instance."""
    problem = TSP.create_small()

    print("=" * 65)
    print("  TSP — Classical Graph Search (small instance, 4 cities)")
    print(f"  Cities: {problem.city_names}")
    print(f"  Distance matrix:")
    for i, name in enumerate(problem.city_names):
        print(f"    {name}: {problem.dist_matrix[i].tolist()}")
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
            _print_tsp_result(info)
            print(f"  Nodes explored: {len(algo.history)}")
        else:
            print("  No solution found.")
            print(f"  Nodes explored: {len(algo.history)}")
        print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Local Search (Hill Climbing)
# ============================================================================

def demo_tsp_local_search():
    """Run Hill Climbing on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Hill Climbing (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = HillClimbingParameter(iteration=500)
    algo = HillClimbing(config, problem)
    initial = problem.sample(1).flatten()
    t0 = time.perf_counter()
    algo.run(initial_state=initial)
    elapsed = time.perf_counter() - t0

    if algo.best_solution is not None:
        best = np.asarray(algo.best_solution).flatten()
        info = problem.decode_permutation(best)
        _print_tsp_result(info)
    else:
        print("  No solution found.")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Simulated Annealing
# ============================================================================

def demo_tsp_sa():
    """Run Simulated Annealing on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Simulated Annealing (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = {
        "initial_temperature": 500.0,
        "cooling_rate": 0.995,
        "min_temperature": 0.01,
        "max_iterations": 5000,
        "n_flips": 1,
    }
    algo = SimulatedAnnealing(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Harmony Search
# ============================================================================

def demo_tsp_hs():
    """Run Harmony Search on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Harmony Search (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
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

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Artificial Bee Colony
# ============================================================================

def demo_tsp_abc():
    """Run Artificial Bee Colony on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Artificial Bee Colony (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = ABCParameter(n_bees=30, limit=50, iteration=500)
    algo = ArtificialBeeColony(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Cuckoo Search
# ============================================================================

def demo_tsp_cs():
    """Run Cuckoo Search on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Cuckoo Search (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = CuckooSearchParameter(n_nests=25, pa=0.25, alpha=0.01, beta=1.5, iteration=500)
    algo = CuckooSearch(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Firefly Algorithm
# ============================================================================

def demo_tsp_fa():
    """Run Firefly Algorithm on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Firefly Algorithm (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = FireflyParameter(
        n_fireflies=25, alpha=0.5, beta0=1.0,
        gamma=1.0, alpha_decay=0.97, cycle=300,
    )
    algo = FireflyAlgorithm(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Ant System
# ============================================================================

def demo_tsp_ant_system():
    """Run Ant System on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Ant System (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = AntSystemParameter(rho=0.5, m=20, q=100.0, alpha=1.0, beta=2.0, cycle=200)
    algo = AntSystem(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — Ant Colony System
# ============================================================================

def demo_tsp_acs():
    """Run Ant Colony System on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — Ant Colony System (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = ACSParameter(rho=0.1, xi=0.1, m=20, q0=0.9, alpha=1.0, beta=2.0, cycle=200)
    algo = ACS(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# TSP — MAX-MIN Ant System
# ============================================================================

def demo_tsp_mmas():
    """Run MAX-MIN Ant System on a TSP instance."""
    problem = TSP.create_medium()

    print("\n" + "=" * 65)
    print("  TSP — MAX-MIN Ant System (medium instance, 6 cities)")
    print(f"  Cities: {problem.city_names}")
    print("=" * 65)

    config = MMASParameter(rho=0.02, m=20, alpha=1.0, beta=3.0, cycle=200)
    algo = MMAS(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_permutation(best)
    _print_tsp_result(info)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("#  TRAVELLING SALESMAN PROBLEM (TSP)")
    print("#" * 65)

    demo_tsp_graph_search()
    demo_tsp_local_search()
    demo_tsp_sa()
    demo_tsp_hs()
    demo_tsp_abc()
    demo_tsp_cs()
    demo_tsp_fa()
    demo_tsp_ant_system()
    demo_tsp_acs()
    demo_tsp_mmas()
