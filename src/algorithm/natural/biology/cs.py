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
    beta : float
        Lévy flight exponent controlling tail heaviness. Typical: 1.5.
    cycle : int
        Number of iterations.
    """

    n_nests: int
    pa: float
    alpha: float
    beta: float
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
    sigma_u: float

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
        self.fitness = cast(np.ndarray, problem.eval(self.nests))

        # Precompute Mantegna's sigma_u (depends only on beta)
        beta = configuration.beta
        num = gamma(1 + beta) * sin(pi * beta / 2)
        den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        self.sigma_u = (num / den) ** (1 / beta)

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.nests[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])

    def _levy_flight(self, shape: tuple[int, ...] | int) -> np.ndarray:
        """Generate Lévy flight steps using Mantegna's algorithm.

        Draws step lengths from a Lévy-stable distribution with the
        configured β exponent, using precomputed sigma_u.

        Parameters
        ----------
        shape : tuple[int, ...] | int
            Shape of the output array — e.g. (n_nests, n_dim) for a
            batch or (n_dim,) for a single step.

        Returns
        -------
        np.ndarray
            Lévy flight step array of the requested shape.
        """
        beta = self.conf.beta

        u = np.random.normal(0, self.sigma_u, size=shape)
        v = np.random.normal(0, 1, size=shape)

        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _clamp(self, solution: np.ndarray) -> np.ndarray:
        """Clamp solution(s) to problem bounds.

        Supports both a single solution of shape (n_dim,) and a batch
        of solutions of shape (n, n_dim).  Bounds are broadcast
        automatically via NumPy.

        Parameters
        ----------
        solution : np.ndarray
            Solution array of shape (n_dim,) or (n, n_dim).

        Returns
        -------
        np.ndarray
            Clamped solution(s) with same shape as input.
        """
        lower = self.problem.bounds[:, 0]  # (n_dim,)
        upper = self.problem.bounds[:, 1]  # (n_dim,)
        return np.clip(solution, lower, upper)

    def generate_cuckoos(self) -> np.ndarray:
        """Generate new cuckoo eggs for every nest via batch Lévy flights.

        Each nest produces one candidate by adding a Lévy-flight step
        scaled by the search-space range per dimension.

        Returns
        -------
        np.ndarray
            New candidate solutions of shape (n_nests, n_dim).
        """
        n = self.conf.n_nests
        steps = self._levy_flight((n, self.n_dim))  # (n_nests, n_dim)

        # Scale step by problem range so displacement is problem-relative

        new_nests = self.nests + self.conf.alpha * steps * (
            self.best_solution - self.nests
        )
        return self._clamp(new_nests)

    def evaluate_cuckoos(self, cuckoos: np.ndarray):
        """Batch-evaluate cuckoos and replace nests where fitness improves.

        Parameters
        ----------
        cuckoos : np.ndarray
            Candidate solutions of shape (n_nests, n_dim).
        """
        cuckoo_fitness = cast(np.ndarray, self.problem.eval(cuckoos))  # (n_nests,)

        # Boolean mask: where the new cuckoo beats the current nest
        improved = cuckoo_fitness < self.fitness
        self.nests[improved] = cuckoos[improved]
        self.fitness[improved] = cuckoo_fitness[improved]

        # Update global best
        best_new_idx = int(np.argmin(cuckoo_fitness))
        if cuckoo_fitness[best_new_idx] < self.best_fitness:
            self.best_fitness = float(cuckoo_fitness[best_new_idx])
            self.best_solution = cuckoos[best_new_idx].copy()

    def abandon_worst_nests(self):
        """Abandon the worst pa fraction of nests (rank-based elitism).

        The top (1 - pa) nests by fitness are kept. The remaining worst
        nests are replaced via biased random walks mixing two random
        existing nests, scaled by a random factor.  All operations are
        fully vectorized.
        """
        n = self.conf.n_nests
        n_abandon = int(n * self.conf.pa)

        if n_abandon == 0:
            return

        # Rank-based selection: abandon the worst nests
        ranked = np.argsort(self.fitness)
        abandon_idx = ranked[-n_abandon:]

        # Biased random walk using two random permutations
        perm1 = np.random.randint(0, n, size=n_abandon)
        perm2 = np.random.randint(0, n, size=n_abandon)
        step_size = np.random.rand(n_abandon, 1)

        new_nests = self.nests[abandon_idx] + step_size * (
            self.nests[perm1] - self.nests[perm2]
        )
        new_nests = self._clamp(new_nests)

        self.nests[abandon_idx] = new_nests

        # Batch evaluate all replaced nests at once
        new_fitness = cast(np.ndarray, self.problem.eval(new_nests))
        self.fitness[abandon_idx] = new_fitness

        # Update global best
        best_new_idx = int(np.argmin(new_fitness))
        if new_fitness[best_new_idx] < self.best_fitness:
            self.best_fitness = float(new_fitness[best_new_idx])
            self.best_solution = new_nests[best_new_idx].copy()

    @override
    def run(self) -> np.ndarray:
        """Execute the Cuckoo Search algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        for _ in range(self.conf.cycle):
            # Batch generate and evaluate cuckoos via Lévy flights
            cuckoos = self.generate_cuckoos()
            self.evaluate_cuckoos(cuckoos)

            # Abandon worst nests
            self.abandon_worst_nests()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
