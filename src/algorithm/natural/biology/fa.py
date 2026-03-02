"""Firefly Algorithm (FA) for continuous optimization.

Inspired by the flashing behaviour of fireflies. Each firefly represents a
candidate solution whose brightness is proportional to its fitness. Fireflies
are attracted to brighter (better) fireflies, and the attractiveness decreases
with distance. A random component is added to avoid premature convergence.

Reference: Yang, X.-S. (2008). Nature-Inspired Metaheuristic Algorithms.
Luniver Press. Chapter 8 — Firefly Algorithms.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


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


class FireflyAlgorithm(
    Model[ContinuousProblem, np.ndarray | None, float, FireflyParameter]
):
    """Firefly Algorithm for continuous optimization.

    Algorithm outline per iteration:
    1. **Pairwise comparison** — For every pair of fireflies *(i, j)*,
       if firefly *j* appears brighter than firefly *i* (comparing
       distance-attenuated light intensity), firefly *i* moves toward
       *j* with an attractiveness that decays exponentially with distance.
    2. **Random perturbation** — A small random step scaled by *alpha*
       is added to each firefly's position to maintain exploration.
    3. **Alpha decay** — The randomness parameter *alpha* is reduced
       by the decay factor to shift from exploration to exploitation
       over time.
    """

    positions: np.ndarray  # shape (n_fireflies, n_dim)
    fitness: np.ndarray  # shape (n_fireflies,)
    light_intensity: np.ndarray  # shape (n_fireflies,) — intrinsic brightness
    n_dim: int
    firefly_pos_history: list[np.ndarray] = []

    def __init__(self, configuration: FireflyParameter, problem: ContinuousProblem):
        """Initialize the Firefly Algorithm.

        Parameters
        ----------
        configuration : FireflyParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        """
        super().__init__(configuration, problem)
        self.name = "Firefly Algorithm"
        self.n_dim = problem.n_dim

        # Initialise firefly positions uniformly within bounds
        self.positions = problem.sample(configuration.n_fireflies)
        self.fitness = cast(np.ndarray, problem.eval(self.positions))
        self.light_intensity = self._compute_light_intensity(self.fitness)

        # Track the global best
        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.positions[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp(self, position: np.ndarray) -> np.ndarray:
        """Clamp a position to the problem bounds.

        Parameters
        ----------
        position : np.ndarray
            Position vector(s).

        Returns
        -------
        np.ndarray
            Clamped position.
        """
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(position, lower, upper)

    @staticmethod
    def _compute_light_intensity(objective_values: np.ndarray) -> np.ndarray:
        """Convert objective (minimisation) values to light intensity.

        Lower objective values yield higher brightness so that better
        solutions emit more light.  Uses the transformation:

        .. math::
            I_0 = \\frac{1}{1 + f} \\quad (f \\ge 0), \\qquad
            I_0 = 1 + |f| \\quad (f < 0)

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

        .. math::
            \\beta(r) = \\beta_0 \\, e^{-\\gamma \\, r^2}

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

        For each pair of fireflies *(i, j)*, the perceived light intensity
        of *j* at firefly *i* is computed as:

        .. math::
            I_j(r_{ij}) = I_{0,j} \\, e^{-\\gamma \\, r_{ij}^2}

        If :math:`I_j(r_{ij}) > I_{0,i}` (firefly *j* appears brighter
        than *i*'s own intrinsic brightness), firefly *i* moves toward *j*:

        .. math::
            x_i \\leftarrow x_i
                + \\beta(r_{ij}) \\, (x_j - x_i)
                + \\alpha \\, (\\text{rand} - 0.5)
                \\, (\\text{upper} - \\text{lower})

        This distance-dependent comparison allows nearby dim fireflies to
        still attract each other, promoting niching and multi-modal search.

        Parameters
        ----------
        alpha : float
            Current randomisation parameter (may be decayed).
        """
        n = self.conf.n_fireflies
        span = self.problem.bounds[:, 1] - self.problem.bounds[:, 0]

        for i in range(n):
            for j in range(n):
                # Perceived intensity of j at the position of i
                diff = self.positions[j] - self.positions[i]
                distance_sq = float(np.sum(diff ** 2))
                perceived_intensity_j = (
                    self.light_intensity[j]
                    * np.exp(-self.conf.gamma * distance_sq)
                )

                if perceived_intensity_j > self.light_intensity[i]:
                    # Distance-dependent attractiveness
                    beta = self._attractiveness(distance_sq)

                    # Move firefly i toward j with random perturbation
                    random_step = (
                        alpha * (np.random.rand(self.n_dim) - 0.5) * span
                    )
                    self.positions[i] += beta * diff + random_step
                    self.positions[i] = self._clamp(self.positions[i])

                    # Re-evaluate fitness and light intensity after move
                    self.fitness[i] = cast(
                        float, self.problem.eval(self.positions[i])
                    )
                    self.light_intensity[i] = float(
                        self._compute_light_intensity(
                            np.array([self.fitness[i]])
                        )[0]
                    )

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
            self.firefly_pos_history.append(self.positions.copy())

        return cast(np.ndarray, self.best_solution)
