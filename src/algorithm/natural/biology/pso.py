"""Particle Swarm Optimization (PSO) for continuous optimization.

Models the social behaviour of bird flocking or fish schooling. Each particle
represents a candidate solution that moves through the search space by
adjusting its velocity according to its own best-known position and the
swarm's global best position.

Reference: Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization.
Proc. IEEE International Conference on Neural Networks, pp. 1942-1948.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


@dataclass
class PSOParameter:
    """Configuration parameters for Particle Swarm Optimization.

    Attributes
    ----------
    n_particles : int
        Number of particles in the swarm. Typical: 20-50.
    w : float
        Inertia weight. Controls the influence of the previous velocity.
        Higher values encourage exploration; lower values favour
        exploitation. Typical: 0.4-0.9.
    c1 : float
        Cognitive (personal) acceleration coefficient. Controls the
        attraction toward a particle's personal best. Typical: 1.5-2.5.
    c2 : float
        Social (global) acceleration coefficient. Controls the attraction
        toward the swarm's global best. Typical: 1.5-2.5.
    v_max : float | None
        Maximum velocity clamp per dimension. If ``None``, the velocity
        is unclamped. Typical: a fraction of the search range.
    cycle : int
        Number of iterations.
    """

    n_particles: int
    w: float
    c1: float
    c2: float
    v_max: float | None
    cycle: int


class ParticleSwarmOptimization(
    Model[ContinuousProblem, np.ndarray | None, float, PSOParameter]
):
    """Particle Swarm Optimization for continuous optimization.

    Algorithm outline per iteration:
    1. **Velocity update** - Each particle's velocity is updated using
       the inertia component, the cognitive component (attraction to
       personal best), and the social component (attraction to global
       best).
    2. **Position update** - Each particle moves according to its new
       velocity.
    3. **Evaluation** - Fitness of each particle is evaluated;
       personal and global bests are updated accordingly.
    """

    positions: np.ndarray    # shape (n_particles, n_dim)
    velocities: np.ndarray   # shape (n_particles, n_dim)
    fitness: np.ndarray      # shape (n_particles,)
    p_best: np.ndarray       # personal best positions
    p_best_fitness: np.ndarray  # personal best fitness values
    n_dim: int

    def __init__(
        self, configuration: PSOParameter, problem: ContinuousProblem
    ):
        """Initialize Particle Swarm Optimization.

        Parameters
        ----------
        configuration : PSOParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        """
        super().__init__(configuration, problem)
        self.name = "Particle Swarm Optimization"
        self.n_dim = problem.n_dim

        # Initialise positions uniformly within bounds
        self.positions = problem.sample(configuration.n_particles)
        self.fitness = cast(np.ndarray, problem.eval(self.positions))

        # Initialise velocities to small random values
        lower = problem.bounds[:, 0]
        upper = problem.bounds[:, 1]
        span = upper - lower
        self.velocities = np.random.uniform(
            -span, span, size=(configuration.n_particles, self.n_dim)
        )

        # Personal bests
        self.p_best = self.positions.copy()
        self.p_best_fitness = self.fitness.copy()

        # Global best
        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.positions[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clamp_position(self, position: np.ndarray) -> np.ndarray:
        """Clamp a position to the problem bounds.

        Parameters
        ----------
        position : np.ndarray
            Position vector(s).

        Returns
        -------
        np.ndarray
            Clamped position.
        """
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]
        return np.clip(position, lower, upper)

    def _clamp_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Clamp velocity to ``[-v_max, v_max]`` if configured.

        Parameters
        ----------
        velocity : np.ndarray
            Velocity vector(s).

        Returns
        -------
        np.ndarray
            Clamped velocity.
        """
        if self.conf.v_max is not None:
            return np.clip(velocity, -self.conf.v_max, self.conf.v_max)
        return velocity

    def _update_best(self):
        """Update personal and global bests after evaluation."""
        improved = self.fitness < self.p_best_fitness
        self.p_best[improved] = self.positions[improved].copy()
        self.p_best_fitness[improved] = self.fitness[improved]

        best_idx = int(np.argmin(self.p_best_fitness))
        if self.p_best_fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.p_best_fitness[best_idx])
            self.best_solution = self.p_best[best_idx].copy()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update_swarm(self):
        """Perform one iteration of the PSO velocity and position update.

        For each particle *i* the velocity is updated as:

        .. math::
            v_i \\leftarrow w \\, v_i
                + c_1 \\, r_1 \\, (p_{\\text{best},i} - x_i)
                + c_2 \\, r_2 \\, (g_{\\text{best}} - x_i)

        where :math:`r_1, r_2 \\sim U(0,1)` are independent random matrices.

        The position is then updated as:

        .. math::
            x_i \\leftarrow x_i + v_i
        """
        n = self.conf.n_particles

        r1 = np.random.rand(n, self.n_dim)
        r2 = np.random.rand(n, self.n_dim)

        assert self.best_solution is not None

        # Velocity update
        cognitive = self.conf.c1 * r1 * (self.p_best - self.positions)
        social = self.conf.c2 * r2 * (self.best_solution - self.positions)
        self.velocities = self.conf.w * self.velocities + cognitive + social
        self.velocities = self._clamp_velocity(self.velocities)

        # Position update
        self.positions = self.positions + self.velocities
        self.positions = self._clamp_position(self.positions)

        # Evaluate new positions
        self.fitness = cast(np.ndarray, self.problem.eval(self.positions))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the Particle Swarm Optimization algorithm.

        Returns
        -------
        np.ndarray
            Best solution found after all cycles.
        """
        for _ in range(self.conf.cycle):
            self.update_swarm()
            self._update_best()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
