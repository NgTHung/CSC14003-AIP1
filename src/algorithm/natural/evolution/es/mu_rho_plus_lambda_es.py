"""(μ/ρ + λ)-ES — Elitist Evolution Strategy with recombination.

An elitist ("plus") strategy where parents survive to compete directly
with their offspring.  This guarantees that the best individual found
so far is never lost, at the cost of potentially slower step-size
adaptation when σ is mismatched.

The algorithm supports intermediate recombination (averaging ρ parents)
and self-adaptive per-component step-sizes via log-normal mutation.

Reference: Beyer, H.-G. & Schwefel, H.-P. (2002). Evolution strategies —
A comprehensive introduction. Natural Computing, 1, 3-52.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class MuRhoPlusLambdaESParameter:
    """Configuration parameters for the (μ/ρ + λ)-ES.

    Attributes
    ----------
    mu : int
        Number of parents (μ).
    rho : int
        Number of parents involved in recombination per offspring.
        ρ = 1 → no recombination; ρ = μ → global intermediate
        recombination.
    lam : int
        Number of offspring generated each generation (λ).
    sigma_init : float
        Initial step-size for every individual. Typical: 1.0.
    cycle : int
        Maximum number of generations.
    """

    mu: int
    rho: int
    lam: int
    sigma_init: float
    cycle: int


class MuRhoPlusLambdaES(
    Model[
        ContinuousProblem,
        np.ndarray | None,
        float,
        MuRhoPlusLambdaESParameter,
    ]
):
    """(μ/ρ + λ)-ES with self-adaptive step-sizes.

    Algorithm per generation:
    1. **Recombination** — For each of λ offspring, average ρ randomly
       chosen parents (both *x* and *σ*).
    2. **Mutation** — Mutate σ via log-normal rule, then perturb x.
    3. **Evaluation** — Compute f(x) for all λ offspring.
    4. **Plus selection** — Pool the μ parents **and** λ offspring;
       keep the best μ individuals for the next generation.
    """

    population: np.ndarray     # shape (mu, n_dim)
    sigmas: np.ndarray         # shape (mu, n_dim)
    fitness: np.ndarray        # shape (mu,)
    n_dim: int

    def __init__(
        self,
        configuration: MuRhoPlusLambdaESParameter,
        problem: ContinuousProblem,
    ):
        if configuration.rho < 1 or configuration.rho > configuration.mu:
            raise ValueError(
                f"ρ must be in [1, μ], got {configuration.rho}"
            )

        super().__init__(configuration, problem)
        self.name = "(μ/ρ+λ)-ES"
        self.n_dim = problem.n_dim

        self.population = problem.sample(configuration.mu)
        self.sigmas = np.full(
            (configuration.mu, self.n_dim), configuration.sigma_init
        )
        self.fitness = np.array(
            [float(cast(np.floating, problem.eval(x))) for x in self.population]
        )

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(x, lower, upper)

    def _recombine(self) -> tuple[np.ndarray, np.ndarray]:
        """Intermediate recombination of ρ random parents."""
        indices = np.random.choice(
            self.conf.mu, size=self.conf.rho, replace=False
        )
        x_rec = np.mean(self.population[indices], axis=0)
        sigma_rec = np.mean(self.sigmas[indices], axis=0)
        return x_rec, sigma_rec

    @override
    def run(self) -> np.ndarray:
        n = self.n_dim
        tau = 1.0 / np.sqrt(2.0 * n)
        sigma_min = 1e-12

        for _ in range(self.conf.cycle):
            offspring_x = np.empty((self.conf.lam, n))
            offspring_sigma = np.empty((self.conf.lam, n))
            offspring_fitness = np.empty(self.conf.lam)

            for k in range(self.conf.lam):
                # 1. Recombination
                x_rec, sigma_rec = self._recombine()

                # 2. Mutate σ (log-normal)
                sigma_new = sigma_rec * np.exp(tau * np.random.randn(n))
                sigma_new = np.maximum(sigma_new, sigma_min)

                # 3. Mutate x
                x_new = x_rec + sigma_new * np.random.randn(n)
                x_new = self._clamp(x_new)

                offspring_x[k] = x_new
                offspring_sigma[k] = sigma_new
                offspring_fitness[k] = float(
                    cast(np.floating, self.problem.eval(x_new))
                )

            # 4. Plus selection — pool parents + offspring, keep best μ
            pool_x = np.vstack([self.population, offspring_x])
            pool_sigma = np.vstack([self.sigmas, offspring_sigma])
            pool_fitness = np.concatenate([self.fitness, offspring_fitness])

            ranking = np.argsort(pool_fitness)
            survivors = ranking[: self.conf.mu]

            self.population = pool_x[survivors]
            self.sigmas = pool_sigma[survivors]
            self.fitness = pool_fitness[survivors]

            # Track best
            if self.fitness[0] < self.best_fitness:
                self.best_fitness = float(self.fitness[0])
                self.best_solution = self.population[0].copy()

            self.history.append(self.best_fitness)

        return cast(np.ndarray, self.best_solution)
