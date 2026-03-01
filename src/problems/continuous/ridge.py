"""Parabolic Ridge benchmark function for continuous optimization.

Tests an algorithm's ability to move along a ridge axis while balancing
the penalty for deviating into other axes.  Progress is made by decreasing
x_1 (unbounded below), but the remaining axes are penalised, creating a
conflict between short-term caution and long-term stride length.

Mathematical form:
    f(x) = x_1 + ξ * (sum(x_i^2 for i=2..n))^(α/2)

Global minimum: f → -∞ as x_1 → -∞ (with x_2=…=x_n=0)
Typical domain: [-5, 5]^n  (bounded for practical testing)
"""

from typing import override

import numpy as np
from .continuous import ContinuousProblem


class Ridge(ContinuousProblem):
    """Parabolic Ridge benchmark function.

    .. math::
        f(x) = x_1 + \\xi \\left( \\sum_{i=2}^{n} x_i^2 \\right)^{\\alpha/2}

    The optimal strategy is to decrease :math:`x_1` as far as possible
    while keeping all other coordinates near zero.  The penalty
    coefficient ``ξ`` and exponent ``α`` control how sharply deviation
    from the ridge axis is punished.

    Attributes
    ----------
    _xi : float
        Penalty coefficient (default: 100).
    _alpha : float
        Penalty exponent (default: 2).
    """

    _xi: float
    _alpha: float

    def __init__(
        self,
        n_dim: int = 2,
        bounds: np.ndarray | None = None,
        *,
        xi: float = 100.0,
        alpha: float = 2.0,
    ):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-5, 5].
        xi : float, optional
            Penalty coefficient (ξ > 0). Default is 100.
        alpha : float, optional
            Penalty exponent (α > 0). Default is 2.
        """
        self._xi = xi
        self._alpha = alpha

        if bounds is None:
            bounds = np.tile(np.array([[-5.0, 5.0]]), (n_dim, 1))
        elif bounds.shape[0] != n_dim:
            raise ValueError(
                f"Bounds shape {bounds.shape} doesn't match n_dim={n_dim}"
            )

        super().__init__("Ridge", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate Parabolic Ridge function.

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
            tail_sq = np.sum(values[1:] ** 2)
            return np.float64(
                values[0] + self._xi * tail_sq ** (self._alpha / 2)
            )

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        tail_sq = np.sum(values[:, 1:] ** 2, axis=1)
        return values[:, 0] + self._xi * tail_sq ** (self._alpha / 2)
