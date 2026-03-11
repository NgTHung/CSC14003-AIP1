"""Demo: test all evolutionary algorithms (GA, DE, ES variants).

Runs every algorithm against Sphere, Ackley, and Rastrigin benchmarks
and reports best fitness + pass/fail status.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from AIP.problems.continuous.sphere import Sphere
from AIP.problems.continuous.ackley import Ackley
from AIP.problems.continuous.rastrigin import Rastrigin

from AIP.algorithm.natural.evolution.ga import GeneticAlgorithm, GAParameter
from AIP.algorithm.natural.evolution.de import DifferentialEvolution, DEParameter
from AIP.algorithm.natural.evolution.es.one_plus_one_es import OnePlusOneES, OnePlusOneESParameter
from AIP.algorithm.natural.evolution.es.mu_rho_plus_lambda_es import MuRhoPlusLambdaES, MuRhoPlusLambdaESParameter
from AIP.algorithm.natural.evolution.es.self_adaptive_es import SelfAdaptiveES, SelfAdaptiveESParameter
from AIP.algorithm.natural.evolution.es.cma_es import CMAES, CMAESParameter


def test_algorithm(name, algo):
    """Run an algorithm and print results."""
    print(f"\n{'='*60}")
    print(f"  {name} on {algo.problem.__class__.__name__}")
    print(f"{'='*60}")
    try:
        result = algo.run()
        print(f"  Best fitness : {algo.best_fitness:.6f}")
        print(f"  Best solution: {np.round(algo.best_solution, 4)}")
        print(f"  Status       : PASSED ✓")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Status       : FAILED ✗ — {e}")


if __name__ == "__main__":
    dim = 10
    problems = [
        Sphere(n_dim=dim),
        Ackley(n_dim=dim),
        Rastrigin(n_dim=dim),
    ]

    for problem in problems:
        print(f"\n{'#'*60}")
        print(f"  Problem: {problem.__class__.__name__} (dim={dim})")
        print(f"{'#'*60}")

        # --- Genetic Algorithm ---
        ga_cfg = GAParameter(pop_size=50, n_bits=16, pc=0.8, pm=0.01, cycle=200)
        ga = GeneticAlgorithm(ga_cfg, problem)
        test_algorithm("Genetic Algorithm (GA)", ga)

        # --- Differential Evolution ---
        de_cfg = DEParameter(pop_size=50, F=0.8, Cr=0.9, cycle=200)
        de = DifferentialEvolution(de_cfg, problem)
        test_algorithm("Differential Evolution (DE)", de)

        # --- (1+1)-ES ---
        one_cfg = OnePlusOneESParameter(sigma=0.5, cycle=1000)
        one_es = OnePlusOneES(one_cfg, problem)
        test_algorithm("(1+1)-ES", one_es)

        # --- (μ/ρ + λ)-ES ---
        mu_cfg = MuRhoPlusLambdaESParameter(mu=15, rho=2, lam=100, sigma_init=1.0, cycle=200)
        mu_es = MuRhoPlusLambdaES(mu_cfg, problem)
        test_algorithm("(μ/ρ + λ)-ES", mu_es)

        # --- Self-Adaptive (μ, λ)-ES ---
        sa_cfg = SelfAdaptiveESParameter(mu=15, lam=100, rho=2, sigma_init=1.0, cycle=200)
        sa_es = SelfAdaptiveES(sa_cfg, problem)
        test_algorithm("Self-Adaptive (μ,λ)-ES", sa_es)

        # --- CMA-ES ---
        cma_cfg = CMAESParameter(sigma=0.5, cycle=200)
        cma = CMAES(cma_cfg, problem)
        test_algorithm("CMA-ES", cma)

    print(f"\n{'='*60}")
    print("  All tests completed!")
    print(f"{'='*60}")