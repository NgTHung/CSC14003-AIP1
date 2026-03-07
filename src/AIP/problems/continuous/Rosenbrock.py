"""Rosenbrock benchmark function for continuous optimization.

Mathematical form:
    f(x) = sum( 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 )

Global minimum: f(1,...,1) = 0
Typical domain: [-5, 10]^n (or [-2.048, 2.048]^n)
"""

from typing import override
import numpy as np
from .continuous import ContinuousProblem


class Rosenbrock(ContinuousProblem):
    """Rosenbrock (banana / valley) benchmark function.

    A classic unimodal test function with a narrow, curved valley
    leading to the global minimum. Easy to find the valley but hard
    to converge to the optimum.

    Formula:
        f(x) = sum( 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2 )

    Global minimum: f(1,...,1) = 0
    Search domain: [-5, 10]^n
    """

    def __init__(self, n_dim: int = 2, bounds: np.ndarray | None = None):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality (must be >= 2). Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-5, 10].
        """
        if n_dim < 2:
            raise ValueError("Rosenbrock requires at least 2 dimensions")

        if bounds is None:
            bounds = np.tile(np.array([[-5.0, 10.0]]), (n_dim, 1))
        elif bounds.shape[0] != n_dim:
            raise ValueError(
                f"Bounds shape {bounds.shape} doesn't match n_dim={n_dim}"
            )

        super().__init__("Rosenbrock", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate the Rosenbrock function.

        Parameters
        ----------
        values : np.ndarray
            Shape (n_dim,) for a single solution or (pop_size, n_dim)
            for a batch.

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
            xi = values[:-1]
            xi1 = values[1:]
            return np.float64(
                np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2)
            )

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        xi = values[:, :-1]
        xi1 = values[:, 1:]
        return np.sum(100.0 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2, axis=1)
