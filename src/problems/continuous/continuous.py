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

    Attributes
    ----------
    bounds : np.ndarray
        The boundaries of the search space.
    n_dim : int
        The dimensionality of the problem (number of decision variables).
    """

    _bounds: np.ndarray
    _n_dim: int

    def __init__(self,
                 name: str,
                 bounds: np.ndarray):
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
        """
        self._bounds = bounds
        self._n_dim = bounds.shape[0]
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
    @property
    def bounds(self):
        """np.ndarray : Search space boundaries, shape (n_dim, 2)."""
        return self._bounds

    @property
    def n_dim(self):
        """int : Number of decision variables."""
        return self._n_dim
