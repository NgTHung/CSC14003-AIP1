"""Abstract base class definitions for continuous optimization problems."""

from typing import override
import numpy as np
from ..base_problem import Problem


class ContinuousProblem(Problem):
    """
    Abstract base class for continuous optimization problems.

    This class extends the base Problem class to
    support continuous decision spaces. The decision
    variables can take any real value within the specified bounds.

    Supports **all** algorithm families through a single class:

    * **Population-based / physics-inspired** — via ``sample()``, ``eval()``,
      ``is_valid()`` inherited from :class:`Problem`.
    * **Local search** (Hill Climbing, Steepest-Ascent) — via
      ``random_state()``, ``neighbors()``, ``value()``, ``is_better()``.

    Sub-classes only need to implement ``eval()``.

    Attributes
    ----------
    bounds : np.ndarray
        The boundaries of the search space.
    n_dim : int
        The dimensionality of the problem (number of decision variables).
    minimize : bool
        Optimization direction. Default True (minimize).
    """

    _bounds: np.ndarray
    _n_dim: int

    def __init__(self,
                 name: str,
                 bounds: np.ndarray,
                 minimize: bool = True,
                 step_size: float = 0.1,
                 n_neighbors: int | None = None):
        """
        Initialize the ContinuousProblem.

        Parameters
        ----------
        name : str
            The name of the problem.
        bounds : np.ndarray
            The boundaries for the decision variables.
            Shape should be (n_dim, 2) where bounds[i, 0] is the lower bound
            and bounds[i, 1] is the upper bound for dimension i.
        minimize : bool, optional
            Optimization direction (default: True = minimize).
        step_size : float, optional
            Fraction of the dimension range used as the perturbation scale
            for ``neighbors()`` (default: 0.1).
        n_neighbors : int or None, optional
            Number of neighbours to generate per call.
            Defaults to ``2 * n_dim`` (one +/- perturbation per dimension).
        """
        self._bounds = bounds
        self._n_dim = bounds.shape[0]
        self.minimize = minimize
        self._step_size = step_size
        self._n_neighbors = n_neighbors if n_neighbors is not None else 2 * self._n_dim
        super().__init__(name)

    @override
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Check if a solution satisfies the problem's boundary constraints.

        Parameters
        ----------
        x : np.ndarray
            The solution vector to check. Shape should be (n_dim,).

        Returns
        -------
        bool
            True if the solution is within all boundary constraints,
            False otherwise.
        """
        if x.shape[0] != self._n_dim:
            return False
        return bool(
            np.all(x >= self._bounds[:, 0]) and np.all(x <= self._bounds[:, 1])
        )

    @override
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """
        Generate random valid solutions within the boundary constraints.

        This method generates uniformly distributed random samples within
        the specified bounds for each dimension.

        Parameters
        ----------
        pop_size : int, optional
            The number of solutions to generate. Default is 1.

        Returns
        -------
        np.ndarray
            An array of randomly generated solutions with shape
            (pop_size, n_dim). Each solution is guaranteed to be
            within the specified bounds.
        """
        lower_bounds = self._bounds[:, 0]
        upper_bounds = self._bounds[:, 1]

        # Generate random samples uniformly distributed in [0, 1]
        random_samples = np.random.rand(pop_size, self._n_dim)

        # Scale and shift to match the bounds
        samples = lower_bounds + random_samples * (upper_bounds - lower_bounds)

        return samples

    # ------------------------------------------------------------------
    # Local-search interface
    # ------------------------------------------------------------------

    def random_state(self) -> np.ndarray:
        """Generate a single random solution within the bounds.

        Returns
        -------
        np.ndarray
            A 1-D solution vector of length ``n_dim``.
        """
        return self.sample(1).flatten()

    def neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """Generate neighbouring solutions by small perturbations.

        For each dimension, the method produces two neighbours: one shifted
        up and one shifted down by ``step_size * range_i``.  The resulting
        position is clipped to remain within the bounds.

        Parameters
        ----------
        state : np.ndarray
            Current solution vector of shape ``(n_dim,)``.

        Returns
        -------
        list[np.ndarray]
            List of neighbour solution vectors.
        """
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        ranges = upper - lower
        nbrs: list[np.ndarray] = []
        for i in range(self._n_dim):
            delta = self._step_size * ranges[i]
            for sign in (+1.0, -1.0):
                nbr = state.copy()
                nbr[i] = np.clip(nbr[i] + sign * delta, lower[i], upper[i])
                nbrs.append(nbr)
                if len(nbrs) >= self._n_neighbors:
                    return nbrs
        return nbrs

    def value(self, state: np.ndarray) -> float:
        """Evaluate the objective for a single solution.

        Parameters
        ----------
        state : np.ndarray
            Solution vector of shape ``(n_dim,)``.

        Returns
        -------
        float
            Objective value (same as ``eval`` for a single solution).
        """
        return float(self.eval(state))

    def is_better(self, value1: float, value2: float) -> bool:
        """Compare two objective values respecting the optimization direction.

        Parameters
        ----------
        value1 : float
            First value.
        value2 : float
            Second value.

        Returns
        -------
        bool
            True if *value1* is strictly better than *value2*.
        """
        return value1 < value2 if self.minimize else value1 > value2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bounds(self):
        """np.ndarray : Search space boundaries, shape (n_dim, 2)."""
        return self._bounds

    @property
    def n_dim(self):
        """int : Number of decision variables."""
        return self._n_dim
