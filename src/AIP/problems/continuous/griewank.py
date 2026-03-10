"""Griewank benchmark function for continuous optimization.

Mathematical form:
    f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i+1)))

Global minimum: f(0,...,0) = 0
Typical domain: [-600, 600]^n
"""

from typing import override
import numpy as np
from .continuous import ContinuousProblem


class Griewank(ContinuousProblem):
    """Griewank multimodal benchmark function.

    Features many widespread local minima regularly distributed,
    but the product term creates a globally smooth structure.
    Becomes easier as dimensionality increases.

    Formula:
        f(x) = 1 + (1/4000)*sum(x_i^2) - prod(cos(x_i / sqrt(i+1)))

    Global minimum: f(0,...,0) = 0
    Search domain: [-600, 600]^n
    """

    def __init__(self, n_dim: int = 2, bounds: np.ndarray | None = None):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-600, 600].
        """
        if bounds is None:
            bounds = np.tile(np.array([[-600.0, 600.0]]), (n_dim, 1))
        elif bounds.shape[0] != n_dim:
            raise ValueError(
                f"Bounds shape {bounds.shape} doesn't match n_dim={n_dim}"
            )

        super().__init__("Griewank", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate the Griewank function.

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
        # Index divisors: sqrt(1), sqrt(2), ..., sqrt(n)
        idx = np.sqrt(np.arange(1, self._n_dim + 1))

        if values.ndim == 1:
            if values.shape[0] != self._n_dim:
                raise ValueError(
                    f"Expected {self._n_dim} dimensions, got {values.shape[0]}"
                )
            sum_sq = np.sum(values ** 2) / 4000.0
            prod_cos = np.prod(np.cos(values / idx))
            return np.float64(1.0 + sum_sq - prod_cos)

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        sum_sq = np.sum(values ** 2, axis=1) / 4000.0
        prod_cos = np.prod(np.cos(values / idx), axis=1)
        return 1.0 + sum_sq - prod_cos
