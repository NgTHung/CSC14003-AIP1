"""Ackley benchmark function for continuous optimization.

Mathematical form:
    f(x) = -a*exp(-b*sqrt(Σxi²/n)) - exp(Σcos(c*xi)/n) + a + e

Global minimum: f(0,...,0) = 0
Typical domain: [-32.768, 32.768]^n
"""

from typing import override

import numpy as np
from ..continuous.continuous import ContinuousProblem


class Ackley(ContinuousProblem):
    """Ackley benchmark function.

    Non-convex test function with many local minima.
    Global minimum: f(0,...,0) = 0

    Attributes
    ----------
    _a, _b, _c : float
        Function parameters (defaults: 20, 0.2, 2π).
    """
    _a: float
    _b: float
    _c: float

    def __init__(
        self,
        n_dim: int = 2,
        bounds: np.ndarray | None = None,
        *,
        a: float = 20,
        b: float = 0.2,
        c: float = 2 * np.pi
    ):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-32.768, 32.768].
        a, b, c : float, optional
            Function parameters. Defaults: a=20, b=0.2, c=2π.
        """
        self._a = a
        self._b = b
        self._c = c

        if bounds is None:
            bounds = np.tile(np.array([[-32.768, 32.768]]), (n_dim, 1))

        super().__init__("Ackley", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate Ackley function.

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
            sum_sq = np.sum(values**2)
            sum_cos = np.sum(np.cos(self._c * values))
            return np.float64(
                -self._a * np.exp(-self._b * np.sqrt(sum_sq / self._n_dim))
                - np.exp(sum_cos / self._n_dim)
                + self._a
                + np.e
            )

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        sum_sq = np.sum(values**2, axis=1)
        sum_cos = np.sum(np.cos(self._c * values), axis=1)
        return (
            -self._a * np.exp(-self._b * np.sqrt(sum_sq / self._n_dim))
            - np.exp(sum_cos / self._n_dim)
            + self._a
            + np.e
        )
