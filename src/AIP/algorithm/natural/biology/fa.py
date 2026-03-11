"""Firefly Algorithm (FA) for continuous and discrete optimization.

Inspired by the flashing behaviour of fireflies. Each firefly represents a
candidate solution whose brightness is proportional to its fitness. Fireflies
are attracted to brighter (better) fireflies, and the attractiveness decreases
with distance. A random component is added to avoid premature convergence.

For **continuous** problems the standard distance-based attractiveness
movement is used. For **discrete** problems (e.g. TSP, Knapsack), when a
firefly is attracted to a brighter one it moves to a random neighbour from
the problem's ``neighbors()`` method instead.

Reference: Yang, X.-S. (2008). Nature-Inspired Metaheuristic Algorithms.
Luniver Press. Chapter 8 — Firefly Algorithms.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from AIP.problems import ContinuousProblem, DiscreteProblem, Problem
from AIP.algorithm import Model


@dataclass
class FireflyParameter:
    """Configuration parameters for the Firefly Algorithm.

    Attributes
    ----------
    n_fireflies : int
        Number of fireflies (population size). Typical: 20-50.
    alpha : float
        Randomisation parameter controlling the step size of the random
        movement component (0 ≤ alpha ≤ 1). Typical: 0.2-0.5.
    beta0 : float
        Base attractiveness at zero distance. Typical: 1.0.
    gamma : float
        Light absorption coefficient controlling how fast attractiveness
        decreases with distance (gamma > 0). Higher values make the
        attraction more localised. Typical: 0.1-10, often set to 1.0.
    alpha_decay : float
        Multiplicative decay factor applied to *alpha* each iteration
        (0 < decay ≤ 1). Setting to 1.0 keeps alpha constant.
        Typical: 0.95-0.99.
    cycle : int
        Number of iterations.
    """

    n_fireflies: int
    alpha: float
    beta0: float
    gamma: float
    alpha_decay: float
    cycle: int


class FireflyAlgorithm(Model[Problem, np.ndarray | None, float, FireflyParameter]):
    """Firefly Algorithm for continuous and discrete optimization.

    Algorithm outline per iteration (following Yang 2008):
    1. **Sequential pairwise comparison** — For every pair *(i, j)*,
       compute the apparent brightness of *j* as seen by *i*:
       $I_j \\cdot e^{-\\gamma r_{ij}^2}$.  If this exceeds the intrinsic
       brightness of *i*, then *i* moves toward *j* and is re-evaluated
       immediately (positions change mid-iteration).
    2. **Random perturbation** — (Continuous only) A small random step
       scaled by *alpha* is added after each attraction move.
    3. **Alpha decay** — *alpha* is reduced by the decay factor each
       iteration to shift from exploration to exploitation over time.
    """

    positions: np.ndarray  # shape (n_fireflies, n_dim)
    fitness: np.ndarray  # shape (n_fireflies,)
    light_intensity: np.ndarray  # shape (n_fireflies,) — intrinsic brightness
    n_dim: int
    _is_continuous: bool
    firefly_pos_history: list[np.ndarray]
    stat: bool

    def __init__(
        self, configuration: FireflyParameter, problem: Problem, stat: bool = False
    ):
        """Initialize the Firefly Algorithm.

        Parameters
        ----------
        configuration : FireflyParameter
            Algorithm hyperparameters.
        problem : Problem
            Optimization problem to solve. Accepts both
            :class:`~problems.ContinuousProblem` and
            :class:`~problems.DiscreteProblem` instances.
        stat : bool, optional
            If True, record full firefly position snapshots each
            iteration into ``firefly_pos_history`` for later analysis
            or visualisation. Defaults to False to save memory.
        """
        super().__init__(configuration, problem)
        self.name = "Firefly Algorithm"
        self._is_continuous = isinstance(problem, ContinuousProblem)
        if self._is_continuous:
            self.n_dim = problem.n_dim  # type: ignore[union-attr]
        else:
            self.n_dim = problem.n_dims  # type: ignore[union-attr]

        # Initialise firefly positions uniformly within bounds
        self.positions = problem.sample(configuration.n_fireflies)
        self.fitness = cast(np.ndarray, problem.eval(self.positions))
        self.light_intensity = self._compute_light_intensity(self.fitness)

        # Track the global best
        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.positions[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

        self.stat = stat
        if stat:
            self.firefly_pos_history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp(self, position: np.ndarray) -> np.ndarray:
        """Clamp a position to the problem bounds (continuous only).

        Parameters
        ----------
        position : np.ndarray
            Position vector(s).

        Returns
        -------
        np.ndarray
            Clamped position.
        """
        problem = cast(ContinuousProblem, self.problem)
        lower = problem.bounds[:, 0]
        upper = problem.bounds[:, 1]
        return np.clip(position, lower, upper)

    @staticmethod
    def _compute_light_intensity(objective_values: np.ndarray) -> np.ndarray:
        """Convert objective (minimisation) values to light intensity.

        Lower objective values yield higher brightness so that better
        solutions emit more light.  Uses the transformation:

            I0 = 1 / (1 + f)    when f >= 0
            I0 = 1 + |f|        when f <  0

        Parameters
        ----------
        objective_values : np.ndarray
            Raw objective/cost values, shape ``(n,)`` or scalar.

        Returns
        -------
        np.ndarray
            Intrinsic light intensity values (higher is better).
        """
        return np.where(
            objective_values >= 0,
            1.0 / (1.0 + objective_values),
            1.0 + np.abs(objective_values),
        )

    def _attractiveness(self, distance_sq: float) -> float:
        """Compute attractiveness given the squared Euclidean distance.

            beta(r) = beta0 * exp(-gamma * r^2)

        Parameters
        ----------
        distance_sq : float
            Squared Euclidean distance between two fireflies.

        Returns
        -------
        float
            Attractiveness value.
        """
        return self.conf.beta0 * np.exp(-self.conf.gamma * distance_sq)

    def _update_best(self):
        """Update the global best solution from the current population."""
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.positions[best_idx].copy()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def move_fireflies(self, alpha: float):
        """Perform one iteration of the Firefly Algorithm movement step.

        Follows Yang (2008): for each pair *(i, j)*, the **apparent**
        brightness of *j* as seen by *i* is compared against *i*'s own
        intrinsic brightness.  If *j* appears brighter, *i* moves toward
        *j* and is **re-evaluated immediately** so that subsequent
        comparisons use the updated position.

        For **continuous** problems the movement is:
            x_i = x_i + beta(r_ij) * (x_j - x_i) + alpha * (rand - 0.5) * span

        For **discrete** problems:
            x_i = random neighbour of x_i  (via problem.random_neighbor())

        Parameters
        ----------
        alpha : float
            Current randomisation parameter (may be decayed).
        """
        n = self.conf.n_fireflies

        if self._is_continuous:
            problem = cast(ContinuousProblem, self.problem)
            span = problem.bounds[:, 1] - problem.bounds[:, 0]

            for i in range(n):
                for j in range(n):
                    # Apparent brightness of j as seen by i
                    diff = self.positions[j] - self.positions[i]
                    dist_sq = float(np.dot(diff, diff))
                    apparent_j = self.light_intensity[j] * np.exp(
                        -self.conf.gamma * dist_sq)

                    if apparent_j > self.light_intensity[i]:
                        beta = self.conf.beta0 * np.exp(
                            -self.conf.gamma * dist_sq)
                        # Move i toward j + random perturbation
                        self.positions[i] += (
                            beta * diff
                            + alpha * (np.random.rand(self.n_dim) - 0.5) * span
                        )
                        self.positions[i] = self._clamp(self.positions[i])

                        # Re-evaluate immediately (sequential update)
                        self.fitness[i] = cast(
                            float, self.problem.eval(self.positions[i]))
                        self.light_intensity[i] = float(
                            self._compute_light_intensity(
                                np.array([self.fitness[i]]))[0])
        else:
            for i in range(n):
                for j in range(n):
                    # For discrete: use intrinsic brightness comparison
                    # (no meaningful Euclidean distance)
                    if self.light_intensity[j] > self.light_intensity[i]:
                        discrete_problem = cast(DiscreteProblem, self.problem)
                        self.positions[i] = discrete_problem.random_neighbor(
                            self.positions[i])

                        self.fitness[i] = cast(
                            float, self.problem.eval(self.positions[i]))
                        self.light_intensity[i] = float(
                            self._compute_light_intensity(
                                np.array([self.fitness[i]]))[0])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the Firefly Algorithm.

        Returns
        -------
        np.ndarray
            Best solution found after all cycles.
        """
        alpha = self.conf.alpha

        for _ in range(self.conf.cycle):
            self.move_fireflies(alpha)
            self._update_best()

            # Decay alpha for gradual transition to exploitation
            alpha *= self.conf.alpha_decay

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

            # For visualization
            if self.stat:
                self.firefly_pos_history.append(self.positions.copy())

        return cast(np.ndarray, self.best_solution)
