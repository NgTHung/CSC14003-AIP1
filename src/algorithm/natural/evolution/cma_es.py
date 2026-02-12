"""Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

The most powerful modern ES variant.  Uses *evolution paths* to learn a
full covariance matrix C that models the pairwise dependencies between
decision variables, enabling efficient search on non-separable and
ill-conditioned landscapes.

Key mechanisms:
- **Cumulative Step-size Adaptation (CSA):** adjusts σ based on the
  conjugate evolution path p_σ, increasing σ when successive steps
  correlate (same direction) and decreasing it when they cancel.
- **Rank-μ covariance update:** learns C from the variance of the
  best-μ mutation steps combined with the rank-one path p_c.

Reference: Hansen, N. & Ostermeier, A. (2001). Completely Derandomized
Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2),
159-195.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class CMAESParameter:
    """Configuration parameters for CMA-ES.

    Most internal constants (c_σ, c_c, c_1, c_μ, damping, weights) are
    derived automatically from μ, λ, and n following Hansen's defaults.
    The user only needs to specify the high-level parameters below.

    Attributes
    ----------
    sigma : float
        Initial global step-size.  Typical: ~1/4 of the search range.
    mu : int | None
        Parent number.  If ``None``, defaults to ``λ // 2``.
    lam : int | None
        Offspring number.  If ``None``, defaults to ``4 + floor(3 ln n)``.
    cycle : int
        Maximum number of generations.
    """

    sigma: float
    cycle: int
    mu: int | None = None
    lam: int | None = None


class CMAES(
    Model[ContinuousProblem, np.ndarray | None, float, CMAESParameter]
):
    """CMA-ES for continuous optimization.

    Algorithm per generation:
    1. **Sampling** — Draw λ offspring from N(m, σ²C).
    2. **Selection & recombination** — Weighted mean of the best μ
       offspring becomes the new distribution mean *m*.
    3. **Update evolution paths** — p_σ (isotropic) and p_c (anisotropic).
    4. **Update σ** via CSA on ||p_σ||.
    5. **Update C** via rank-one (p_c) and rank-μ (successful steps).
    """

    mean: np.ndarray          # distribution mean m, shape (n,)
    sigma: float              # global step-size
    C: np.ndarray             # covariance matrix (n, n)
    p_sigma: np.ndarray       # isotropic evolution path (n,)
    p_c: np.ndarray           # anisotropic evolution path (n,)
    n_dim: int

    # Internal strategy parameters (computed once)
    _mu: int
    _lam: int
    _weights: np.ndarray      # recombination weights (mu,)
    _mu_eff: float
    _c_sigma: float
    _d_sigma: float
    _c_c: float
    _c_1: float
    _c_mu: float
    _chi_n: float             # E[||N(0,I)||]

    def __init__(
        self, configuration: CMAESParameter, problem: ContinuousProblem
    ):
        super().__init__(configuration, problem)
        self.name = "CMA-ES"
        self.n_dim = n = problem.n_dim

        # ── Population sizes ──
        self._lam = configuration.lam or (4 + int(3 * np.log(n)))
        self._mu = configuration.mu or self._lam // 2

        # ── Recombination weights (log-linear, normalised) ──
        raw_w = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        self._weights = raw_w / np.sum(raw_w)
        self._mu_eff = 1.0 / np.sum(self._weights ** 2)

        # ── Strategy parameter defaults (Hansen 2001) ──
        self._c_sigma = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._d_sigma = (
            1
            + 2 * max(0.0, np.sqrt((self._mu_eff - 1) / (n + 1)) - 1)
            + self._c_sigma
        )
        self._c_c = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._c_1 = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        self._c_mu = min(
            1 - self._c_1,
            2 * (self._mu_eff - 2 + 1 / self._mu_eff)
            / ((n + 2) ** 2 + self._mu_eff),
        )
        self._chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # ── State initialisation ──
        self.mean = problem.sample(1)[0]
        self.sigma = configuration.sigma
        self.C = np.eye(n)
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)

        self.best_fitness = float(cast(np.floating, problem.eval(self.mean)))
        self.best_solution = self.mean.copy()
        self.history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp(self, x: np.ndarray) -> np.ndarray:
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(x, lower, upper)

    def _sqrt_C(self) -> np.ndarray:
        """Compute C^{1/2} via eigen-decomposition.

        Returns
        -------
        np.ndarray
            Matrix square-root of C, shape (n, n).
        """
        eigenvalues, B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)  # numerical safety
        D = np.diag(np.sqrt(eigenvalues))
        return B @ D @ B.T

    def _inv_sqrt_C(self) -> np.ndarray:
        """Compute C^{-1/2} via eigen-decomposition."""
        eigenvalues, B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        D_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        return B @ D_inv @ B.T

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        n = self.n_dim

        for _ in range(self.conf.cycle):
            sqrt_C = self._sqrt_C()
            inv_sqrt_C = self._inv_sqrt_C()

            # 1. Sampling — x_k = m + σ C^{1/2} z_k
            z = np.random.randn(self._lam, n)                 # N(0, I)
            y = z @ sqrt_C.T                                   # C^{1/2} z
            offspring = self.mean + self.sigma * y
            for k in range(self._lam):
                offspring[k] = self._clamp(offspring[k])

            # 2. Evaluate & rank
            fitness = np.array(
                [float(cast(np.floating, self.problem.eval(x)))
                 for x in offspring]
            )
            ranking = np.argsort(fitness)

            # 3. Selection & recombination — new mean
            old_mean = self.mean.copy()
            selected = offspring[ranking[: self._mu]]
            self.mean = self._weights @ selected               # weighted average

            # Weighted step in N(0, I) space
            y_w = (self.mean - old_mean) / self.sigma
            z_w = inv_sqrt_C @ y_w

            # 4a. Update isotropic path p_σ
            c_s = self._c_sigma
            self.p_sigma = (
                (1 - c_s) * self.p_sigma
                + np.sqrt(c_s * (2 - c_s) * self._mu_eff) * z_w
            )

            # 4b. Update σ via CSA
            self.sigma *= np.exp(
                (c_s / self._d_sigma)
                * (np.linalg.norm(self.p_sigma) / self._chi_n - 1)
            )

            # 4c. h_σ flag (stall indicator)
            h_sigma = int(
                np.linalg.norm(self.p_sigma)
                / np.sqrt(1 - (1 - c_s) ** (2 * (_ + 1)))
                < (1.4 + 2 / (n + 1)) * self._chi_n
            )

            # 4d. Update anisotropic path p_c
            c_c = self._c_c
            self.p_c = (
                (1 - c_c) * self.p_c
                + h_sigma * np.sqrt(c_c * (2 - c_c) * self._mu_eff) * y_w
            )

            # 5. Covariance matrix update (rank-one + rank-μ)
            # Rank-one
            rank_one = np.outer(self.p_c, self.p_c)

            # Rank-μ
            selected_y = (selected - old_mean) / self.sigma    # (mu, n)
            rank_mu = np.zeros((n, n))
            for i in range(self._mu):
                rank_mu += self._weights[i] * np.outer(
                    selected_y[i], selected_y[i]
                )

            # Correction for h_σ == 0
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)

            self.C = (
                (1 - self._c_1 - self._c_mu + delta_h * self._c_1) * self.C
                + self._c_1 * rank_one
                + self._c_mu * rank_mu
            )

            # Enforce symmetry
            self.C = (self.C + self.C.T) / 2

            # Track best
            best_gen_idx = ranking[0]
            if fitness[best_gen_idx] < self.best_fitness:
                self.best_fitness = float(fitness[best_gen_idx])
                self.best_solution = offspring[best_gen_idx].copy()

            self.history.append(self.best_fitness)

        return cast(np.ndarray, self.best_solution)
