"""Cuckoo Search (CS) algorithm for continuous and discrete optimization.

Inspired by the brood parasitism of cuckoo birds and Lévy flight foraging
patterns. Each egg (nest) represents a solution, and new solutions are
generated via Lévy flights. A fraction of the worst nests are abandoned
and replaced each generation.

For **continuous** problems, Lévy flights are used to generate new cuckoos
and biased random walks to replace abandoned nests.
For **discrete** problems (e.g. TSP, Knapsack), a random neighbour from
the problem's ``neighbors()`` method is used instead, preserving solution
validity.

Reference: Yang, X.-S. & Deb, S. (2009). Cuckoo search via Lévy flights.
Proc. World Congress on Nature & Biologically Inspired Computing, pp. 210-214.
"""

from dataclasses import dataclass
from typing import cast, override
from math import gamma, sin, pi

import numpy as np
from AIP.problems import ContinuousProblem, DiscreteProblem, Problem
from AIP.algorithm import Algorithm


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
    iteration : int
        Number of iterations.
    """

    n_nests: int
    pa: float
    alpha: float
    beta: float
    iteration: int


class CuckooSearch(Algorithm[Problem, np.ndarray | None, float, CuckooSearchParameter]):
    """Cuckoo Search for continuous and discrete optimization.

    Algorithm outline per iteration:
    1. Generate a new cuckoo egg (Lévy flight for continuous, random
       neighbour for discrete) from each nest.
    2. Replace the nest if the new egg is better.
    3. Abandon fraction pa of the worst nests and replace with new
       random solutions.
    """

    nests: np.ndarray
    fitness: np.ndarray
    n_dim: int
    sigma_u: float
    _is_continuous: bool
    nests_history: list[np.ndarray]
    stat: bool

    def __init__(
        self, configuration: CuckooSearchParameter, problem: Problem, stat: bool = False
    ):
        """Initialize Cuckoo Search.

        Parameters
        ----------
        configuration : CuckooSearchParameter
            Algorithm hyperparameters.
        problem : Problem
            Optimization problem to solve. Accepts both
            :class:`~problems.ContinuousProblem` and
            :class:`~problems.DiscreteProblem` instances.
        stat : bool, optional
            If True, record full nest snapshots each iteration into
            ``nests_history`` for later analysis or visualisation.
            Defaults to False to save memory.
        """
        super().__init__(configuration, problem)
        self._is_continuous = isinstance(problem, ContinuousProblem)
        if self._is_continuous:
            self.n_dim = problem.n_dim  # type: ignore[union-attr]
        else:
            self.n_dim = problem.n_dims  # type: ignore[union-attr]
        self.nests = problem.sample(configuration.n_nests)
        self.fitness = cast(np.ndarray, problem.eval(self.nests))

        # Precompute Mantegna's sigma_u for continuous Lévy flights
        if self._is_continuous:
            beta = configuration.beta
            num = gamma(1 + beta) * sin(pi * beta / 2)
            den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
            self.sigma_u = (num / den) ** (1 / beta)
        else:
            self.sigma_u = 0.0  # unused for discrete

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.nests[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

        self.stat = stat
        if stat:
            self.nests_history = []

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
        assert isinstance(self.problem, ContinuousProblem)
        lower = self.problem.bounds[:, 0]  # (n_dim,)
        upper = self.problem.bounds[:, 1]  # (n_dim,)
        return np.clip(solution, lower, upper)

    def _generate_cuckoo_discrete(self, idx: int) -> np.ndarray:
        """Neighbour-based candidate for discrete problems.

        Delegates to the problem's ``neighbors()`` method and selects
        one at random, preserving solution validity.

        Parameters
        ----------
        idx : int
            Index of the current nest.

        Returns
        -------
        np.ndarray
            A random neighbour of the current nest.
        """
        problem = cast(DiscreteProblem, self.problem)
        nbrs = problem.random_neighbor(self.nests[idx])
        return nbrs

    def generate_cuckoos(self) -> np.ndarray:
        """Generate new cuckoo eggs for every nest.

        For continuous problems, uses batch Lévy flights scaled by the
        distance to the global best. For discrete problems, generates
        a random neighbour for each nest.

        Returns
        -------
        np.ndarray
            New candidate solutions of shape (n_nests, n_dim).
        """
        if not self._is_continuous:
            return np.array(
                [self._generate_cuckoo_discrete(i) for i in range(self.conf.n_nests)]
            )

        n = self.conf.n_nests
        steps = self._levy_flight((n, self.n_dim))  # (n_nests, n_dim)

        # Scale step by problem range so displacement is problem-relative
        new_nests = self.nests + self.conf.alpha * steps * (
            self.best_solution - self.nests
        )
        return self._clamp(new_nests)

    def evaluate_cuckoos(self, cuckoos: np.ndarray):
        """Evaluate cuckoos and compare each against a randomly chosen nest.

        Per Yang & Deb (2009): a cuckoo egg generated from nest *i* is
        evaluated against a **randomly chosen** nest *j* (not necessarily
        *i*). If the cuckoo is better, it replaces nest *j*.

        Parameters
        ----------
        cuckoos : np.ndarray
            Candidate solutions of shape (n_nests, n_dim).
        """
        n = self.conf.n_nests
        cuckoo_fitness = cast(np.ndarray, self.problem.eval(cuckoos))  # (n_nests,)

        # Each cuckoo is compared against a randomly chosen nest
        targets = np.random.randint(0, n, size=n)

        for i in range(n):
            j = targets[i]
            if cuckoo_fitness[i] < self.fitness[j]:
                self.nests[j] = cuckoos[i].copy()
                self.fitness[j] = cuckoo_fitness[i]

        # Update global best
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.nests[best_idx].copy()

    def abandon_worst_nests(self):
        """Abandon the worst pa fraction of nests (rank-based elitism).

        The top (1 - pa) nests by fitness are kept. For continuous
        problems the remaining worst nests are replaced via biased
        random walks mixing two random existing nests. For discrete
        problems they are replaced by fresh random samples from
        the problem's ``sample()`` method.
        """
        n = self.conf.n_nests
        n_abandon = int(n * self.conf.pa)

        if n_abandon == 0:
            return

        # Rank-based selection: abandon the worst nests
        ranked = np.argsort(self.fitness)
        abandon_idx = ranked[-n_abandon:]

        if self._is_continuous:
            # Biased random walk using two random permutations
            perm1 = np.random.randint(0, n, size=n_abandon)
            perm2 = np.random.randint(0, n, size=n_abandon)
            step_size = np.random.rand(n_abandon, 1)

            new_nests = self.nests[abandon_idx] + step_size * (
                self.nests[perm1] - self.nests[perm2]
            )
            new_nests = self._clamp(new_nests)
        else:
            # Discrete: replace with fresh random solutions
            new_nests = self.problem.sample(n_abandon)

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
        for _ in range(self.conf.iteration):
            # Batch generate and evaluate cuckoos via Lévy flights
            cuckoos = self.generate_cuckoos()
            self.evaluate_cuckoos(cuckoos)

            # Abandon worst nests
            self.abandon_worst_nests()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())
            if self.stat:
                self.nests_history.append(self.nests.copy())

        return cast(np.ndarray, self.best_solution)
