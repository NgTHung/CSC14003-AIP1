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

    Algorithm outline per iteration:
    1. **Pairwise comparison** — For every pair of fireflies *(i, j)*,
       if firefly *j* appears brighter than firefly *i* (comparing
       distance-attenuated light intensity), firefly *i* moves toward
       *j*. For continuous problems this uses attractiveness-weighted
       movement; for discrete problems a random neighbour is selected.
    2. **Random perturbation** — (Continuous only) A small random step
       scaled by *alpha* is added to maintain exploration.
    3. **Alpha decay** — The randomness parameter *alpha* is reduced
       by the decay factor to shift from exploration to exploitation
       over time.
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

        For each pair of fireflies *(i, j)*, if firefly *j* is intrinsically
        brighter than firefly *i*, firefly *i* moves toward *j* with
        distance-dependent attractiveness.

        For **continuous** problems the movement is vectorised:
            x[i] += sum_j{ beta(r_ij) * (x[j] - x[i]) } + alpha * (rand - 0.5) * span
        where the sum is over all *j* brighter than *i*.

        For **discrete** problems:
            x[i] = random neighbour of x[i]  (via problem.neighbors())

        Parameters
        ----------
        alpha : float
            Current randomisation parameter (may be decayed).
        """
        n = self.conf.n_fireflies

        if self._is_continuous:
            problem = cast(ContinuousProblem, self.problem)
            span = problem.bounds[:, 1] - problem.bounds[:, 0]

            # Pairwise differences: diff[i, j] = positions[j] - positions[i]
            diff = (self.positions[np.newaxis, :, :]
                    - self.positions[:, np.newaxis, :])          # (n, n, d)

            # Pairwise squared distances
            dist_sq = np.sum(diff ** 2, axis=2)                  # (n, n)

            # Attractiveness matrix: beta(r) = beta0 * exp(-gamma * r^2)
            beta = self.conf.beta0 * np.exp(
                -self.conf.gamma * dist_sq)                      # (n, n)

            # Mask: mask[i, j] = True if j is brighter than i
            mask = (self.light_intensity[np.newaxis, :]
                    > self.light_intensity[:, np.newaxis])       # (n, n)

            # Weighted displacement from all brighter fireflies
            displacement = np.sum(
                (beta * mask)[:, :, np.newaxis] * diff, axis=1)  # (n, d)

            # Random perturbation
            random_step = alpha * (
                np.random.rand(n, self.n_dim) - 0.5) * span     # (n, d)

            self.positions += displacement + random_step
            self.positions = self._clamp(self.positions)

            # Batch evaluate once
            self.fitness = cast(np.ndarray, self.problem.eval(self.positions))
            self.light_intensity = self._compute_light_intensity(self.fitness)
        else:
            for i in range(n):
                moved = False
                for j in range(n):
                    if self.light_intensity[j] > self.light_intensity[i]:
                        discrete_problem = cast(DiscreteProblem, self.problem)
                        nbrs = discrete_problem.random_neighbor(self.positions[i])
                        self.positions[i] = nbrs

                        self.fitness[i] = cast(
                            float, self.problem.eval(self.positions[i]))
                        self.light_intensity[i] = float(
                            self._compute_light_intensity(
                                np.array([self.fitness[i]]))[0])
                        moved = True

                if not moved:
                    discrete_problem = cast(DiscreteProblem, self.problem)
                    nbrs = discrete_problem.random_neighbor(self.positions[i])
                    self.positions[i] = nbrs
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
