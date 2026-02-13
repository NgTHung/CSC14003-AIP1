"""Self-Adaptive (μ, λ)-ES with log-normal step-size mutation.

Each individual carries its own step-size σ as part of its genome.
Step-sizes evolve alongside the solution vector: σ is mutated first
(log-normal), then used to perturb x.  Comma selection (parents are
discarded; only the best μ of λ offspring survive) ensures that
poorly-adapted σ values die out naturally.

Reference: Schwefel, H.-P. (1995). Evolution and Optimum Seeking.
Wiley, New York.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class SelfAdaptiveESParameter:
    """Configuration parameters for the self-adaptive (μ, λ)-ES.

    Attributes
    ----------
    mu : int
        Number of parents (μ). Typical: λ // 4.
    lam : int
        Number of offspring (λ). Must satisfy λ > μ for comma selection.
    rho : int
        Number of parents involved in recombination for each offspring.
        ρ = 1 means no recombination (asexual); ρ = μ means global
        intermediate recombination. Typical: 2 or μ.
    sigma_init : float
        Initial step-size for every individual. Typical: 1.0.
    cycle : int
        Maximum number of generations.
    """

    mu: int
    lam: int
    rho: int
    sigma_init: float
    cycle: int


class SelfAdaptiveES(
    Model[ContinuousProblem, np.ndarray | None, float, SelfAdaptiveESParameter]
):
    """Self-Adaptive (μ, λ)-ES with intermediate recombination.

    Algorithm per generation:
    1. **Recombination** — For each of λ offspring, select ρ parents
       uniformly and compute the component-wise mean of their *x* and
       *σ* vectors.
    2. **Mutation of σ** — Log-normal perturbation:
       σ' = σ_recomb · exp(τ · N(0, 1)), with learning rate
       τ = 1 / √(2n).
    3. **Mutation of x** — x' = x_recomb + σ' · N(0, I).
    4. **Selection (comma)** — Discard parents; keep the best μ
       offspring to form the next parent population.
    """

    population: np.ndarray     # shape (mu, n_dim)
    sigmas: np.ndarray         # shape (mu, n_dim) — per-component σ
    fitness: np.ndarray        # shape (mu,)
    n_dim: int

    def __init__(
        self, configuration: SelfAdaptiveESParameter, problem: ContinuousProblem
    ):
        if configuration.lam <= configuration.mu:
            raise ValueError(
                f"λ ({configuration.lam}) must be > μ ({configuration.mu}) "
                "for comma selection."
            )
        if configuration.rho < 1 or configuration.rho > configuration.mu:
            raise ValueError(
                f"ρ must be in [1, μ], got {configuration.rho}"
            )

        super().__init__(configuration, problem)
        self.name = "Self-Adaptive (μ,λ)-ES"
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
        """Intermediate recombination of ρ random parents.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_recomb, σ_recomb) — averaged solution and step-size.
        """
        indices = np.random.choice(self.conf.mu, size=self.conf.rho, replace=False)
        x_recomb = np.mean(self.population[indices], axis=0)
        sigma_recomb = np.mean(self.sigmas[indices], axis=0)
        return x_recomb, sigma_recomb

    @override
    def run(self) -> np.ndarray:
        tau = 1.0 / np.sqrt(2.0 * self.n_dim)
        sigma_min = 1e-12  # floor to prevent σ collapse

        for _ in range(self.conf.cycle):
            offspring_x = np.empty((self.conf.lam, self.n_dim))
            offspring_sigma = np.empty((self.conf.lam, self.n_dim))
            offspring_fitness = np.empty(self.conf.lam)

            for k in range(self.conf.lam):
                # 1. Recombination
                x_rec, sigma_rec = self._recombine()

                # 2. Mutate σ first (log-normal, per-component)
                sigma_new = sigma_rec * np.exp(tau * np.random.randn(self.n_dim))
                sigma_new = np.maximum(sigma_new, sigma_min)

                # 3. Mutate x using new σ
                x_new = x_rec + sigma_new * np.random.randn(self.n_dim)
                x_new = self._clamp(x_new)

                offspring_x[k] = x_new
                offspring_sigma[k] = sigma_new
                offspring_fitness[k] = float(
                    cast(np.floating, self.problem.eval(x_new))
                )

            # 4. Comma selection — best μ offspring become parents
            ranking = np.argsort(offspring_fitness)
            survivors = ranking[: self.conf.mu]

            self.population = offspring_x[survivors]
            self.sigmas = offspring_sigma[survivors]
            self.fitness = offspring_fitness[survivors]

            # Track best
            if self.fitness[0] < self.best_fitness:
                self.best_fitness = float(self.fitness[0])
                self.best_solution = self.population[0].copy()

            self.history.append(self.best_fitness)

        return cast(np.ndarray, self.best_solution)
