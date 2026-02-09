"""Artificial Bee Colony (ABC) algorithm for continuous optimization.

Models the foraging behaviour of honeybee colonies. The colony is divided into
three groups: employed bees, onlooker bees, and scout bees. Employed bees
exploit known food sources, onlooker bees select sources based on fitness-
proportional probability, and scouts replace exhausted sources with random ones.

Reference: Karaboga, D. (2005). An Idea Based on Honey Bee Swarm for
Numerical Optimization. Technical Report TR06, Erciyes University.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


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
    cycle : int
        Number of iterations (foraging cycles).
    """

    n_bees: int
    limit: int
    cycle: int


class ArtificialBeeColony(
    Model[ContinuousProblem, np.ndarray | None, float, ABCParameter]
):
    """Artificial Bee Colony for continuous optimization.

    Algorithm outline per cycle:
    1. **Employed bee phase** - Each employed bee generates a new candidate
       near its current food source and keeps it if it is better (greedy).
    2. **Onlooker bee phase** - Onlooker bees select food sources with
       probability proportional to fitness quality, then perform the same
       local search as employed bees.
    3. **Scout bee phase** - If a food source has not improved for *limit*
       trials, the employed bee becomes a scout and is placed at a random
       position in the search space.
    """

    food_sources: np.ndarray   # shape (n_bees, n_dim)
    fitness: np.ndarray        # shape (n_bees,)
    fit_values: np.ndarray     # transformed fitness for probability calc
    trials: np.ndarray         # stagnation counters per source
    n_dim: int

    def __init__(
        self, configuration: ABCParameter, problem: ContinuousProblem
    ):
        """Initialize the Artificial Bee Colony.

        Parameters
        ----------
        configuration : ABCParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        """
        super().__init__(configuration, problem)
        self.n_dim = problem.n_dim
        self.food_sources = problem.sample(configuration.n_bees)
        self.fitness = cast(np.ndarray, problem.eval(self.food_sources))
        self.fit_values = self._calculate_fit(self.fitness)
        self.trials = np.zeros(configuration.n_bees, dtype=int)

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.food_sources[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        """Clamp a solution to the problem bounds.

        Parameters
        ----------
        solution : np.ndarray
            Solution vector(s).

        Returns
        -------
        np.ndarray
            Clamped solution.
        """
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(solution, lower, upper)

    def _generate_candidate(self, idx: int) -> np.ndarray:
        """Generate a candidate solution near food source *idx*.

        A random dimension *j* is selected and perturbed using a randomly
        chosen partner source *k* (k ≠ idx):

        .. math::
            v_{ij} = x_{ij} + \\phi_{ij} (x_{ij} - x_{kj})

        where :math:`\\phi_{ij} \\in [-1, 1]`.

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

        # Pick a random dimension to mutate
        j = np.random.randint(0, self.n_dim)

        # Pick a random partner (different from idx)
        k = idx
        while k == idx:
            k = np.random.randint(0, self.conf.n_bees)

        phi = np.random.uniform(-1, 1)
        candidate[j] = (
            self.food_sources[idx, j]
            + phi * (self.food_sources[idx, j] - self.food_sources[k, j])
        )
        return self._clamp(candidate)

    def _update_best(self):
        """Update the global best solution from the current population."""
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.food_sources[best_idx].copy()

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

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
                self.fit_values[i] = self._calculate_fit(
                    np.array([candidate_fitness])
                )[0]
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
                self.fit_values[i] = self._calculate_fit(
                    np.array([candidate_fitness])
                )[0]
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
                self.fitness[i] = cast(
                    float, self.problem.eval(self.food_sources[i])
                )
                self.fit_values[i] = self._calculate_fit(
                    np.array([self.fitness[i]])
                )[0]
                self.trials[i] = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the Artificial Bee Colony algorithm.

        Returns
        -------
        np.ndarray
            Best solution found after all cycles.
        """
        for _ in range(self.conf.cycle):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self._update_best()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
