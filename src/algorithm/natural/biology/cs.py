"""Cuckoo Search (CS) algorithm for continuous optimization.

Inspired by the brood parasitism of cuckoo birds and Lévy flight foraging
patterns. Each egg (nest) represents a solution, and new solutions are
generated via Lévy flights. A fraction of the worst nests are abandoned
and replaced each generation.

Reference: Yang, X.-S. & Deb, S. (2009). Cuckoo search via Lévy flights.
Proc. World Congress on Nature & Biologically Inspired Computing, pp. 210-214.
"""

from dataclasses import dataclass
from typing import cast, override
from math import gamma, sin, pi

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class CuckooSearchParameter:
    """Configuration parameters for Cuckoo Search.

    Attributes
    ----------
    n_nests : int
        Number of host nests (population size). Typical: 15-40.
    pa : float
        Discovery probability (0 < pa < 1). Fraction of worst nests
        abandoned per generation. Typical: 0.25.
    alpha : float
        Step size scaling factor for Lévy flights. Typical: 0.01.
    cycle : int
        Number of iterations.
    """

    n_nests: int
    pa: float
    alpha: float
    cycle: int


class CuckooSearch(
    Model[ContinuousProblem, np.ndarray | None, float, CuckooSearchParameter]
):
    """Cuckoo Search for continuous optimization.

    Algorithm outline per iteration:
    1. Generate a new cuckoo egg via Lévy flight from a random nest.
    2. Replace a random nest if the new egg is better.
    3. Abandon fraction pa of the worst nests and replace with new random ones.
    """

    nests: np.ndarray
    fitness: np.ndarray
    n_dim: int

    def __init__(
        self, configuration: CuckooSearchParameter, problem: ContinuousProblem
    ):
        """Initialize Cuckoo Search.

        Parameters
        ----------
        configuration : CuckooSearchParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        """
        super().__init__(configuration, problem)
        self.n_dim = problem.n_dim
        self.nests = problem.sample(configuration.n_nests)
        self.fitness = np.array(
            [cast(float, problem.eval(nest)) for nest in self.nests]
        )

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.nests[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = []

    def _levy_flight(self, size: int) -> np.ndarray:
        """Generate a Lévy flight step using Mantegna's algorithm.

        Draws step lengths from a Lévy-stable distribution with index β = 1.5
        using the ratio of two normal distributions.

        Parameters
        ----------
        size : int
            Dimensionality of the step vector.

        Returns
        -------
        np.ndarray
            Lévy flight step vector of shape (size,).
        """
        beta = 1.5

        # Mantegna's algorithm: sigma for numerator
        num = gamma(1 + beta) * sin(pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma_u = (num / den) ** (1 / beta)

        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)

        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _clamp(self, solution: np.ndarray) -> np.ndarray:
        """Clamp solution to problem bounds.

        Parameters
        ----------
        solution : np.ndarray
            Solution vector of shape (n_dim,).

        Returns
        -------
        np.ndarray
            Clamped solution.
        """
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(solution, lower, upper)

    def get_cuckoo(self) -> np.ndarray:
        """Generate a new cuckoo egg via Lévy flight from a random nest.

        Returns
        -------
        np.ndarray
            New candidate solution of shape (n_dim,).
        """
        i = np.random.randint(0, self.conf.n_nests)
        step = self._levy_flight(self.n_dim)

        # Scale step relative to difference from best solution
        assert self.best_solution is not None
        new_nest = self.nests[i] + self.conf.alpha * step * (
            self.nests[i] - self.best_solution
        )
        return self._clamp(new_nest)

    def evaluate_cuckoo(self, cuckoo: np.ndarray):
        """Evaluate a new cuckoo and replace a random nest if better.

        Parameters
        ----------
        cuckoo : np.ndarray
            Candidate solution of shape (n_dim,).
        """
        cuckoo_fitness = cast(float, self.problem.eval(cuckoo))

        # Compare against a random nest
        j = np.random.randint(0, self.conf.n_nests)
        if cuckoo_fitness < self.fitness[j]:
            self.nests[j] = cuckoo
            self.fitness[j] = cuckoo_fitness

            if cuckoo_fitness < self.best_fitness:
                self.best_fitness = cuckoo_fitness
                self.best_solution = cuckoo.copy()

    def abandon_worst_nests(self):
        """Abandon fraction pa of worst nests and replace with new random ones.

        Biased random walks: new nests are generated by mixing two random
        existing nests, scaled by a random factor, applied only to nests
        selected for abandonment.
        """
        for idx in range(self.conf.n_nests):
            e = np.random.random()
            if self.conf.pa < e:
                continue
            # Generate new nest via biased random walk
            r1, r2 = np.random.choice(self.conf.n_nests, size=2, replace=False)
            step_size = np.random.random()
            new_nest = self.nests[idx] + step_size * (
                self.nests[r1] - self.nests[r2]
            )
            new_nest = self._clamp(new_nest)
            new_fitness = cast(float, self.problem.eval(new_nest))

            self.nests[idx] = new_nest
            self.fitness[idx] = new_fitness

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_nest.copy()

    @override
    def run(self) -> np.ndarray:
        """Execute the Cuckoo Search algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        for _ in range(self.conf.cycle):
            # Generate new cuckoo via Lévy flight and evaluate
            cuckoo = self.get_cuckoo()
            self.evaluate_cuckoo(cuckoo)

            # Abandon worst nests
            self.abandon_worst_nests()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
