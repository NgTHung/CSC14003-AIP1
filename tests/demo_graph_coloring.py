"""Demo script for the Graph Coloring Problem with multiple algorithm families.

Demonstrates:
1. Classical graph search (DFS, BFS, UCS, Greedy, A*)
2. Local search (Hill Climbing)
3. Physics-inspired (Simulated Annealing, Harmony Search)
4. Biology-inspired (Artificial Bee Colony, Cuckoo Search, Firefly Algorithm)

Note: ACO algorithms construct permutations (each node visited once), which is
incompatible with graph coloring where values must be in [0, n_colors).
Evolution-based and human-inspired algorithms are continuous-only.
"""

import time
import numpy as np
from AIP.problems.discrete.graph_coloring import GraphColoring
from AIP.algorithm.classical.DFS import DepthFirstSearch
from AIP.algorithm.classical.BFS import BreadthFirstSearch
from AIP.algorithm.classical.UCS import UniformCostSearch
from AIP.algorithm.classical.GreedyBestFirst import GreedyBestFirstSearch
from AIP.algorithm.classical.AStar import AStarSearch
from AIP.algorithm.local.HillClimbing import HillClimbing, HillClimbingParameter
from AIP.algorithm.natural.physic.SA import (
    SimulatedAnnealing,
    SimulatedAnnealingParameter,
)
from AIP.algorithm.natural.physic.HS import HarmonySearch, HarmonySearchParameter
from AIP.algorithm.natural.biology.abc import ArtificialBeeColony, ABCParameter
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter


def _print_coloring_result(info: dict, problem: GraphColoring, indent: str = "  "):
    """Print a decoded graph coloring solution."""
    print(f"{indent}Coloring       : {info['coloring']}")
    print(f"{indent}Named          : {info['coloring_named']}")
    print(f"{indent}Conflicts      : {info['n_conflicts']}")
    print(f"{indent}Legal coloring : {info['is_legal']}")
    print(f"{indent}Colors used    : {len(set(info['coloring']))}/{problem.n_colors}")


def demo_gc_graph_search():
    """Run classical graph search algorithms on a small Graph Coloring instance."""
    problem = GraphColoring.create_small()

    print("=" * 65)
    print("  Graph Coloring — Classical Graph Search (small instance)")
    print(f"  Vertices : {problem.n_vertices}  {problem.vertex_names}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {problem.edges}")
    print("=" * 65)

    config: dict = {}
    algorithms = [
        ("DFS", DepthFirstSearch(config, problem)),
        ("BFS", BreadthFirstSearch(config, problem)),
        ("UCS", UniformCostSearch(config, problem)),
        ("Greedy BFS", GreedyBestFirstSearch(config, problem)),
        ("A*", AStarSearch(config, problem)),
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
            _print_coloring_result(info, problem)
            print(f"  Nodes explored: {len(algo.history)}")
        else:
            print("  No solution found.")
            print(f"  Nodes explored: {len(algo.history)}")
        print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_local_search():
    """Run Hill Climbing on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Hill Climbing (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
    print("=" * 65)

    config = HillClimbingParameter(iteration=500)
    algo = HillClimbing(config, problem)
    initial = problem.sample(1).flatten()
    t0 = time.perf_counter()
    algo.run(initial_state=initial)
    elapsed = time.perf_counter() - t0

    if algo.best_solution is not None:
        best = np.asarray(algo.best_solution).flatten()
        info = problem.decode_coloring(best)
        _print_coloring_result(info, problem)
    else:
        print("  No solution found.")
    print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_sa():
    """Run Simulated Annealing on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Simulated Annealing (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
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

    info = problem.decode_coloring(best)
    _print_coloring_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_hs():
    """Run Harmony Search on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Harmony Search (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
    print("=" * 65)

    config = HarmonySearchParameter(
        hms= 30,
        hmcr= 0.9,
        par= 0.3,
        bw= 0.1,
        max_iterations= 3000,
    )
    algo = HarmonySearch(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_coloring(best)
    _print_coloring_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_abc():
    """Run Artificial Bee Colony on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Artificial Bee Colony (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
    print("=" * 65)

    config = ABCParameter(n_bees=30, limit=50, iteration=500)
    algo = ArtificialBeeColony(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_coloring(best)
    _print_coloring_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_cs():
    """Run Cuckoo Search on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Cuckoo Search (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
    print("=" * 65)

    config = CuckooSearchParameter(
        n_nests=25, pa=0.25, alpha=0.01, beta=1.5, iteration=500
    )
    algo = CuckooSearch(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_coloring(best)
    _print_coloring_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


def demo_gc_fa():
    """Run Firefly Algorithm on a Graph Coloring instance."""
    problem = GraphColoring.create_medium()

    print("\n" + "=" * 65)
    print("  Graph Coloring — Firefly Algorithm (Petersen graph)")
    print(f"  Vertices : {problem.n_vertices}")
    print(f"  Colors   : {problem.n_colors}")
    print(f"  Edges    : {len(problem.edges)}")
    print("=" * 65)

    config = FireflyParameter(
        n_fireflies=25,
        alpha=0.5,
        beta0=1.0,
        gamma=1.0,
        alpha_decay=0.97,
        cycle=300,
    )
    algo = FireflyAlgorithm(config, problem)
    t0 = time.perf_counter()
    best = algo.run()
    elapsed = time.perf_counter() - t0

    info = problem.decode_coloring(best)
    _print_coloring_result(info, problem)
    print(f"  Iterations   : {len(algo.history)}")
    print(f"  Runtime      : {elapsed:.4f}s")


if __name__ == "__main__":
    print("\n" + "#" * 65)
    print("#  GRAPH COLORING PROBLEM")
    print("#" * 65)

    demo_gc_graph_search()
    demo_gc_local_search()
    demo_gc_sa()
    demo_gc_hs()
    demo_gc_abc()
    demo_gc_cs()
    demo_gc_fa()
