"""Sphere benchmark function for continuous optimization.

Mathematical form:
    f(x) = sum(x_i^2)

Global minimum: f(0,...,0) = 0
Typical domain: [-5.12, 5.12]^n
"""

from typing import override
import numpy as np
from .continuous import ContinuousProblem


class Sphere(ContinuousProblem):
    """Sphere benchmark function.

    Unimodal convex test function.
    Global minimum: f(0,...,0) = 0
    """

    def __init__(self, n_dim: int = 2, bounds: np.ndarray | None = None):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-5.12, 5.12].
        """
        if bounds is None:
            bounds = np.tile(np.array([[-5.12, 5.12]]), (n_dim, 1))
        elif bounds.shape[0] != n_dim:
            raise ValueError(
                f"Bounds shape {bounds.shape} doesn't match n_dim={n_dim}"
            )

        super().__init__("Sphere", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate Sphere function.

        Parameters
        ----------
        values : np.ndarray
            Shape (n_dim,) or (pop_size, n_dim).

        Returns
        -------
        np.float64 or np.ndarray
            Fitness value(s).
        """
        if values.ndim == 1:
            if values.shape[0] != self._n_dim:
                raise ValueError(
                    f"Expected {self._n_dim} dimensions, got {values.shape[0]}"
                )
            return np.sum(values ** 2)
        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        return np.sum(values ** 2, axis=1)