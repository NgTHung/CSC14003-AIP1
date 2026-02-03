"""Rastrigin benchmark function for continuous optimization.

Mathematical form:
    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))

Global minimum: f(0,...,0) = 0
Typical domain: [-5.12, 5.12]^n
"""

from typing import override
import numpy as np
from .continuous import ContinuousProblem


class Rastrigin(ContinuousProblem):
    """
    Rastrigin multimodal benchmark function.

    Formula: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0
    Search domain: [-5.12, 5.12]^n

    Attributes
    ----------
    _a : float
        Amplitude parameter (default: 10).
    """

    _a: float

    def __init__(
        self,
        n_dim: int = 2,
        bounds: np.ndarray | None = None,
        *,
        A: float = 10,
    ):
        """
        Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-5.12, 5.12].
        A : float, optional
            Amplitude parameter. Default is 10.
        """
        self._a = A

        # Set default bounds if not provided
        if bounds is None:
            bounds = np.tile(np.array([[-5.12, 5.12]]), (n_dim, 1))
        else:
            assert bounds.shape[0] == n_dim

        super().__init__("Rastrigin", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """
        Evaluate the Rastrigin function.

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
            return np.float64(
                self._a * self._n_dim
                + np.sum(values**2 - self._a * np.cos(2 * np.pi * values))
            )

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        return self._a * self._n_dim + np.sum(
            values**2 - self._a * np.cos(2 * np.pi * values), axis=1
        )
