"""(1+1)-ES with the 1/5th Success Rule.

The simplest Evolution Strategy: a single parent produces a single offspring
via isotropic Gaussian mutation.  The step-size σ is adapted on-line using
Rechenberg's 1/5th success rule — if too many mutations succeed the step is
enlarged; if too few succeed the step is shrunk.

Reference: Rechenberg, I. (1973). Evolutionsstrategie: Optimierung
technischer Systeme nach Prinzipien der biologischen Evolution.
Frommann-Holzboog.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class OnePlusOneESParameter:
    """Configuration parameters for the (1+1)-ES.

    Attributes
    ----------
    sigma : float
        Initial step-size (mutation strength). Typical: 1.0.
    cycle : int
        Maximum number of generations.
    adaptation_interval : int
        Number of generations between σ adjustments.  The success rate
        is measured over the last *adaptation_interval* mutations.
        Typical: 10 * n_dim.
    a : float
        Step-size update factor (a > 1).  σ is multiplied by *a* when the
        success rate exceeds 1/5, and divided by *a* otherwise.
        Typical: 1.5 (Schwefel) or 1/0.85 ≈ 1.176 (Rechenberg).
    """

    sigma: float
    cycle: int
    adaptation_interval: int = 0   # 0 → will be set to 10*n_dim
    a: float = 1.5


class OnePlusOneES(
    Model[ContinuousProblem, np.ndarray | None, float, OnePlusOneESParameter]
):
    """(1+1)-ES with the 1/5th success rule.

    Algorithm per generation:
    1. Mutate the parent with isotropic Gaussian noise scaled by σ.
    2. If the offspring is at least as good, it replaces the parent and
       the mutation is counted as a success.
    3. Every *adaptation_interval* generations, adjust σ:
       - success rate > 1/5 → σ *= a  (too easy, explore wider)
       - success rate < 1/5 → σ /= a  (too hard, exploit locally)
    """

    solution: np.ndarray
    sigma: float
    n_dim: int

    def __init__(
        self, configuration: OnePlusOneESParameter, problem: ContinuousProblem
    ):
        super().__init__(configuration, problem)
        self.name = "(1+1)-ES"
        self.n_dim = problem.n_dim
        self.sigma = configuration.sigma

        if configuration.adaptation_interval <= 0:
            self.conf.adaptation_interval = 10 * self.n_dim

        self.solution = problem.sample(1)[0]
        self.best_fitness = float(cast(np.floating, problem.eval(self.solution)))
        self.best_solution = self.solution.copy()
        self.history = []

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(x, lower, upper)

    @override
    def run(self) -> np.ndarray:
        n_success = 0
        interval = self.conf.adaptation_interval

        for gen in range(self.conf.cycle):
            # Mutation: isotropic Gaussian
            offspring = self.solution + self.sigma * np.random.randn(self.n_dim)
            offspring = self._clamp(offspring)

            offspring_fitness = float(cast(np.floating, self.problem.eval(offspring)))

            # Plus selection: offspring replaces parent if at least as good
            if offspring_fitness <= self.best_fitness:
                self.solution = offspring
                self.best_fitness = offspring_fitness
                self.best_solution = offspring.copy()
                n_success += 1

            # 1/5th rule adaptation
            if (gen + 1) % interval == 0:
                success_rate = n_success / interval
                if success_rate > 1 / 5:
                    self.sigma *= self.conf.a      # increase step
                elif success_rate < 1 / 5:
                    self.sigma /= self.conf.a      # decrease step
                # success_rate == 1/5 → keep σ unchanged
                n_success = 0

            self.history.append(self.best_fitness)

        return cast(np.ndarray, self.best_solution)
