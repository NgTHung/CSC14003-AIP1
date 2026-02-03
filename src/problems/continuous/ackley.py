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
    a, b, c : float
        Function parameters (defaults: 20, 0.2, π)
    """
    __a: float
    __b: float
    __c: float

    def __init__(
        self,
        bounds: np.ndarray = np.array([[-32768, 32768], [-32768, 32768]]),
        *,
        a: float = 20,
        b: float = 0.2,
        c: float = np.pi
    ):
        """Initialize Ackley function.

        Parameters
        ----------
        bounds : np.ndarray
            Search space bounds, shape (n_dim, 2)
        a, b, c : float
            Function parameters (standard: a=20, b=0.2, c=2π)
        """
        self.__a = a
        self.__b = b
        self.__c = c
        super().__init__("Ackley", True, bounds)

    @override
    def eval(self, values: np.ndarray) -> float | np.ndarray:
        """Evaluate Ackley function.

        Parameters
        ----------
        values : np.ndarray
            Solution vector, shape (n_dim,)

        Returns
        -------
        float
            Function value
        """
        assert values.size == self._n_dim
        part_1 = -self.__b * np.sqrt(
            1/self._n_dim * np.sum(np.square(values))
        )
        part_2 = 1/self._n_dim * np.sum(
            np.cos(self.__c * values)
        )
        return (
            -self.__a * np.exp(part_1)
            - np.exp(part_2)
            + self.__a
            + np.exp(1)
        )