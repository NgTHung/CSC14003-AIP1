"""Artificial Bee Colony (ABC) algorithm for continuous and discrete optimization.

Models the foraging behaviour of honeybee colonies. The colony is divided into
three groups: employed bees, onlooker bees, and scout bees. Employed bees
exploit known food sources, onlooker bees select sources based on fitness-
proportional probability, and scouts replace exhausted sources with random ones.

For **continuous** problems the standard arithmetic perturbation is used.
For **discrete** problems (e.g. TSP) a two-position swap operator is used
instead, which preserves solution validity.

Reference: Karaboga, D. (2005). An Idea Based on Honey Bee Swarm for
Numerical Optimization. Technical Report TR06, Erciyes University.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from AIP.problems import ContinuousProblem, DiscreteProblem, Problem
from AIP.algorithm import Algorithm


@dataclass
class ABCParameter:
    """Configuration parameters for the Artificial Bee Colony algorithm.

    Attributes
    ----------
    n_bees : int
        Number of food sources (employed bees). The total colony size is
        ``2 * n_bees`` (employed + onlooker). Typical: 20-50.
    limit : int
        Abandonment limit. A food source that has not been improved for
        *limit* consecutive trials is replaced by a scout bee.
        Common default: ``n_bees * n_dim / 2``.
    iteration : int
        Number of iterations (foraging cycles).
    """

    n_bees: int
    limit: int
    iteration: int


class ArtificialBeeColony(Algorithm[Problem, np.ndarray | None, float, ABCParameter]):
    """Artificial Bee Colony for continuous and discrete optimization.

    Algorithm outline per iteration:
    1. **Employed bee phase** - Each employed bee generates a new candidate
       near its current food source and keeps it if it is better (greedy).
    2. **Onlooker bee phase** - Onlooker bees select food sources with
       probability proportional to fitness quality, then perform the same
       local search as employed bees.
    3. **Scout bee phase** - If a food source has not improved for *limit*
       trials, the employed bee becomes a scout and is placed at a random
       position in the search space.

    For **continuous** problems the standard arithmetic perturbation is used
    to generate candidates.  For **discrete** problems a two-position swap
    operator is applied instead, preserving solution feasibility.
    """

    food_sources: np.ndarray  # shape (n_bees, n_dim)
    fitness: np.ndarray  # shape (n_bees,)
    fit_values: np.ndarray  # transformed fitness for probability calc
    trials: np.ndarray  # stagnation counters per source
    n_dim: int
    _is_continuous: bool
    food_sources_history: list[np.ndarray]
    stat: bool

    @override
    def reset(self):
        self._is_continuous = isinstance(self.problem, ContinuousProblem)
        if self._is_continuous:
            self.n_dim = self.problem.n_dim  # type: ignore[union-attr]
        else:
            self.n_dim = self.problem.n_dims  # type: ignore[union-attr]
        self.food_sources = self.problem.sample(self.conf.n_bees)
        self.fitness = cast(np.ndarray, self.problem.eval(self.food_sources))
        self.fit_values = self._calculate_fit(self.fitness)
        self.trials = np.zeros(self.conf.n_bees, dtype=int)

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.food_sources[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []
        if self.stat:
            self.food_sources_history = []

    def __init__(
        self, configuration: ABCParameter, problem: Problem, stat: bool = False
    ):
        """Initialize the Artificial Bee Colony.

        Parameters
        ----------
        configuration : ABCParameter
            Algorithm hyperparameters.
        problem : Problem
            Optimization problem to solve. Accepts both
            :class:`~problems.ContinuousProblem` and
            :class:`~problems.DiscreteProblem` instances.
        stat : bool, optional
            If True, record full population snapshots each iteration
            into ``food_sources_history`` for later analysis or
            visualisation. Defaults to False to save memory.
        """
        self.stat = stat
        self.conf = configuration
        super().__init__(configuration, problem)

    @staticmethod
    def _calculate_fit(objective_values: np.ndarray) -> np.ndarray:
        """Convert objective (minimisation) values to non-negative fitness.

        Uses the standard ABC transformation so that lower objective values
        yield higher fitness values for the roulette-wheel selection.

        Parameters
        ----------
        objective_values : np.ndarray
            Raw objective/cost values, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Transformed fitness values, shape ``(n,)``.
        """
        fit = np.where(
            objective_values >= 0,
            1.0 / (1.0 + objective_values),
            1.0 + np.abs(objective_values),
        )
        return fit

    def _clamp(self, solution: np.ndarray) -> np.ndarray:
        """Clamp a solution to the problem bounds (continuous problems only).

        Parameters
        ----------
        solution : np.ndarray
            Solution vector(s).

        Returns
        -------
        np.ndarray
            Clamped solution.
        """
        problem = cast(ContinuousProblem, self.problem)
        lower = problem.bounds[:, 0]
        upper = problem.bounds[:, 1]
        return np.clip(solution, lower, upper)

    def _generate_candidate_continuous(self, idx: int) -> np.ndarray:
        """Arithmetic perturbation candidate for continuous problems.

        A random dimension *j* is selected and perturbed using a randomly
        chosen partner source *k* (k ≠ idx):

            v[i,j] = x[i,j] + phi * (x[i,j] - x[k,j])

        where phi is drawn uniformly from [-1, 1].

        Parameters
        ----------
        idx : int
            Index of the current food source.

        Returns
        -------
        np.ndarray
            New candidate solution of shape ``(n_dim,)``.
        """
        candidate = self.food_sources[idx].copy()

        j = np.random.randint(0, self.n_dim)

        k = idx
        while k == idx:
            k = np.random.randint(0, self.conf.n_bees)

        phi = np.random.uniform(-1, 1)
        candidate[j] = self.food_sources[idx, j] + phi * (
            self.food_sources[idx, j] - self.food_sources[k, j]
        )
        return self._clamp(candidate)

    def _generate_candidate_discrete(self, idx: int) -> np.ndarray:
        """Neighbour-based candidate for discrete problems.

        Delegates to the problem's ``neighbors()`` method and selects
        one at random.  This keeps ABC generic — the neighbourhood
        structure (bit-flip for Knapsack, 2-opt for TSP, etc.) is
        defined by the problem itself.

        Parameters
        ----------
        idx : int
            Index of the current food source.

        Returns
        -------
        np.ndarray
            A random neighbour of the current food source.
        """
        problem = cast(DiscreteProblem, self.problem)
        nbrs = problem.random_neighbor(self.food_sources[idx])
        return nbrs

    def _generate_candidate(self, idx: int) -> np.ndarray:
        """Generate a candidate solution near food source *idx*.

        Delegates to the continuous (arithmetic) or discrete (swap)
        strategy depending on the problem type.

        Parameters
        ----------
        idx : int
            Index of the current food source.

        Returns
        -------
        np.ndarray
            New candidate solution of shape ``(n_dim,)``.
        """
        if self._is_continuous:
            return self._generate_candidate_continuous(idx)
        return self._generate_candidate_discrete(idx)

    def _update_best(self):
        """Update the global best solution from the current population."""
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.food_sources[best_idx].copy()

    def employed_bee_phase(self):
        """Employed bee phase: local search around each food source.

        Each employed bee generates a candidate near its food source.
        If the candidate is better, the source is replaced and the trial
        counter is reset; otherwise the trial counter is incremented.
        """
        for i in range(self.conf.n_bees):
            candidate = self._generate_candidate(i)
            candidate_fitness = cast(float, self.problem.eval(candidate))

            if candidate_fitness < self.fitness[i]:
                self.food_sources[i] = candidate
                self.fitness[i] = candidate_fitness
                self.fit_values[i] = self._calculate_fit(np.array([candidate_fitness]))[
                    0
                ]
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bee_phase(self):
        """Onlooker bee phase: fitness-proportional source selection.

        Onlooker bees select food sources with probability proportional to
        their fitness. For each selected source the same local-search
        operator as in the employed phase is applied.
        """
        # Roulette-wheel probabilities
        total_fit = np.sum(self.fit_values)
        if total_fit == 0:
            probs = np.ones(self.conf.n_bees) / self.conf.n_bees
        else:
            probs = self.fit_values / total_fit

        for _ in range(self.conf.n_bees):
            # Select a food source proportional to fitness
            i = int(np.random.choice(self.conf.n_bees, p=probs))

            candidate = self._generate_candidate(i)
            candidate_fitness = cast(float, self.problem.eval(candidate))

            if candidate_fitness < self.fitness[i]:
                self.food_sources[i] = candidate
                self.fitness[i] = candidate_fitness
                self.fit_values[i] = self._calculate_fit(np.array([candidate_fitness]))[
                    0
                ]
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def scout_bee_phase(self):
        """Scout bee phase: abandon exhausted food sources.

        Any food source whose trial counter exceeds the *limit* threshold
        is discarded and replaced by a uniformly random solution within the
        search bounds.
        """
        for i in range(self.conf.n_bees):
            if self.trials[i] > self.conf.limit:
                self.food_sources[i] = self.problem.sample(1)[0]
                self.fitness[i] = cast(float, self.problem.eval(self.food_sources[i]))
                self.fit_values[i] = self._calculate_fit(np.array([self.fitness[i]]))[0]
                self.trials[i] = 0

    @override
    def run(self) -> np.ndarray:
        """Execute the Artificial Bee Colony algorithm.

        Returns
        -------
        np.ndarray
            Best solution found after all cycles.
        """
        self.reset()
        for _ in range(self.conf.iteration):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self._update_best()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())
            if self.stat:
                self.food_sources_history.append(self.food_sources.copy())

        return cast(np.ndarray, self.best_solution)
