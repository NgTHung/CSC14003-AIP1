"""Cigar benchmark function for continuous optimization.

Tests an algorithm's ability to handle ill-conditioned problems where
variables have vastly different scales of influence on the objective.

Mathematical form:
    f(x) = x_1^2 + ξ * sum(x_i^2 for i=2..n)

When ξ is large the function forms a long, narrow valley resembling a cigar.
Without covariance-matrix adaptation (e.g. CMA-ES) standard algorithms
struggle significantly on this landscape.

Global minimum: f(0, ..., 0) = 0
Typical domain: [-5, 5]^n
"""

from typing import override

import numpy as np
from .continuous import ContinuousProblem


class Cigar(ContinuousProblem):
    """Cigar (ill-conditioned ellipsoid) benchmark function.

    .. math::
        f(x) = x_1^2 + \\xi \\sum_{i=2}^{n} x_i^2

    The condition number ``ξ`` controls the eccentricity of the iso-cost
    contours.  Larger values make the problem harder for algorithms
    that cannot adapt different step sizes per direction.

    Attributes
    ----------
    _xi : float
        Condition number (default: 1e6).
    """

    _xi: float

    def __init__(
        self,
        n_dim: int = 2,
        bounds: np.ndarray | None = None,
        *,
        xi: float = 1e6,
    ):
        """Parameters
        ----------
        n_dim : int, optional
            Problem dimensionality. Default is 2.
        bounds : np.ndarray or None, optional
            Custom bounds (n_dim, 2). If None, uses [-5, 5].
        xi : float, optional
            Condition number (ξ ≥ 1). Default is 1 000 000.
        """
        self._xi = xi

        if bounds is None:
            bounds = np.tile(np.array([[-5.0, 5.0]]), (n_dim, 1))
        elif bounds.shape[0] != n_dim:
            raise ValueError(
                f"Bounds shape {bounds.shape} doesn't match n_dim={n_dim}"
            )

        super().__init__("Cigar", bounds)

    @override
    def eval(self, values: np.ndarray) -> np.float64 | np.ndarray:
        """Evaluate Cigar function.

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
                values[0] ** 2 + self._xi * np.sum(values[1:] ** 2)
            )

        if values.shape[1] != self._n_dim:
            raise ValueError(
                f"Expected {self._n_dim} dimensions, got {values.shape[1]}"
            )
        return values[:, 0] ** 2 + self._xi * np.sum(values[:, 1:] ** 2, axis=1)
