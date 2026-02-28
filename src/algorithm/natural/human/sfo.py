"""Social Force Optimization (SFO) for continuous optimization.

Simulates pedestrian crowd dynamics where each agent (pedestrian) updates
its velocity and position under three forces:

  1. **Inertia**          – preserves the current direction of travel,
                            preventing erratic direction changes.
  2. **Attraction Force** – pulls the agent towards the global best
                            position found so far (exploitation).
  3. **Social Force**     – guides the agent towards the mean position
                            of the whole crowd (exploration).

Velocity / position update equations (vectorized over the full population):

    V_new = w * V_old
          + c_attract * r1 * (GlobalBest    - X)
          + c_social  * r2 * (MeanPosition  - X)

    X_new = X + V_new

where ``r1``, ``r2`` ~ Uniform(0, 1) with shape ``(pop_size, n_dim)``.

The inertia weight *w* is reduced each iteration via a multiplicative
decay ``w ← w * 0.99`` (configurable as ``w_decay``), transitioning the
swarm from broad exploration towards fine-grained exploitation.

Reference: Inspired by Helbing & Molnár (1995) social force model and
subsequent adaptations to metaheuristic optimisation.
"""

from dataclasses import dataclass

import numpy as np

from algorithm.base_model import Model
from problems.continuous.continuous import ContinuousProblem


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SFOConfig:
    """Hyperparameters for the Social Force Optimization algorithm.

    Attributes
    ----------
    pop_size : int
        Number of agents (pedestrians) in the crowd.  Typical: 30–100.
    iterations : int
        Maximum number of update cycles to run.
    minimization : bool
        ``True`` to minimise the objective; ``False`` to maximise it.
    w : float
        Initial inertia weight.  Controls how much of the previous
        velocity is retained.  Decays each iteration by ``w_decay``.
        Typical starting value: 0.9.
    w_decay : float
        Multiplicative decay factor applied to *w* each iteration
        (``w ← w * w_decay``).  A value of 0.99 gives a gradual linear-
        style decay over ~300 iterations before the weight becomes small.
        Default: 0.99.
    c_attract : float
        Attraction coefficient – strength of the pull towards the global
        best position.  Default: 1.5.
    c_social : float
        Social coefficient – strength of the pull towards the crowd mean.
        Default: 1.5.
    """

    pop_size: int   = 50
    iterations: int = 200
    minimization: bool = True
    w: float        = 0.9
    w_decay: float  = 0.99
    c_attract: float = 1.5
    c_social: float  = 1.5


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

class SFO(Model[ContinuousProblem, np.ndarray, float, SFOConfig]):
    """Social Force Optimization (SFO) algorithm.

    Each iteration, the entire population is updated simultaneously using
    fully vectorized NumPy operations — there are **no Python-level loops
    over individual agents**.

    Attributes
    ----------
    population : np.ndarray, shape (pop_size, n_dim)
        Current agent positions in the search space.
    velocity : np.ndarray, shape (pop_size, n_dim)
        Current velocities of all agents.
    fitness : np.ndarray, shape (pop_size,)
        Fitness values corresponding to ``population``.
    best_solution : np.ndarray, shape (n_dim,)
        Best position found over the complete run.
    best_fitness : float
        Fitness value of ``best_solution``.
    history : list[float]
        Best-fitness value recorded at the end of each iteration,
        suitable for plotting the convergence curve.
    """

    name: str = "Social Force Optimization"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        """Set up population, velocities, fitness array, and global best.

        Called once at the start of :meth:`run`.  Positions are drawn via
        the problem's own :meth:`~ContinuousProblem.sample` method so that
        they are guaranteed to lie within the search bounds.
        """
        prob = self.problem
        cfg  = self.conf

        # --- Positions (pop_size × n_dim) ---
        self.population = prob.sample(cfg.pop_size)

        # --- Velocities: start at rest ---
        # Initialising to zero is stable; the social and attraction forces
        # generate movement from the very first iteration.
        self.velocity = np.zeros((cfg.pop_size, prob.n_dim))

        # --- Fitness of the initial population (vectorized eval) ---
        self.fitness = self.problem.eval(self.population).astype(float)

        # --- History ---
        self.history = []

        # --- Establish the initial global best ---
        self._update_global_best()

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Execute the SFO optimisation loop.

        Returns
        -------
        np.ndarray, shape (n_dim,)
            The best solution found across all iterations.
        """
        self._initialise()

        cfg = self.conf
        lb  = self.problem._bounds[:, 0]   # shape (n_dim,)
        ub  = self.problem._bounds[:, 1]   # shape (n_dim,)

        # Mutable inertia weight; will be decayed each iteration.
        w = cfg.w

        for _ in range(cfg.iterations):

            # ==============================================================
            # 1.  SOCIAL COMPONENT – crowd mean position
            # ==============================================================
            # mean_pos: shape (n_dim,).  NumPy broadcasts it to
            # (pop_size, n_dim) automatically in the velocity equation.
            mean_pos = np.mean(self.population, axis=0)

            # ==============================================================
            # 2.  STOCHASTIC COEFFICIENTS  (pop_size × n_dim)
            # ==============================================================
            # Each agent receives an independent random draw, maintaining
            # diversity across the entire population in a single operation.
            r1 = np.random.rand(cfg.pop_size, self.problem.n_dim)
            r2 = np.random.rand(cfg.pop_size, self.problem.n_dim)

            # ==============================================================
            # 3.  VELOCITY UPDATE  (fully vectorized – zero Python loops)
            # ==============================================================
            # V_new = w * V_old
            #       + c_attract * r1 * (GlobalBest - X)   ← exploitation
            #       + c_social  * r2 * (Mean       - X)   ← exploration
            #
            # self.best_solution: shape (n_dim,)  →  broadcast to (pop_size, n_dim)
            # self.population:    shape (pop_size, n_dim)
            self.velocity = (
                w * self.velocity
                + cfg.c_attract * r1 * (self.best_solution - self.population)
                + cfg.c_social  * r2 * (mean_pos           - self.population)
            )

            # ==============================================================
            # 4.  POSITION UPDATE  (vectorized)
            # ==============================================================
            new_population = self.population + self.velocity

            # ==============================================================
            # 5.  BOUNDARY ENFORCEMENT  via np.clip  (vectorized)
            # ==============================================================
            # lb / ub broadcast over the population axis automatically,
            # keeping every agent strictly inside the feasible region.
            new_population = np.clip(new_population, lb, ub)

            # ==============================================================
            # 6.  FITNESS EVALUATION  (one vectorized call for all agents)
            # ==============================================================
            new_fitness = self.problem.eval(new_population).astype(float)

            # ==============================================================
            # 7.  GREEDY SELECTION (ELITISM)  – boolean mask, no loops
            # ==============================================================
            # improved: shape (pop_size,) bool – True where the new
            # position is strictly better than the agent's current one.
            if cfg.minimization:
                improved = new_fitness < self.fitness   # lower is better
            else:
                improved = new_fitness > self.fitness   # higher is better

            # Accept only the improved positions; others stay unchanged.
            # Both array writes operate on the full matrices simultaneously.
            self.population[improved] = new_population[improved]
            self.fitness[improved]    = new_fitness[improved]

            # ==============================================================
            # 8.  GLOBAL BEST UPDATE & HISTORY RECORDING
            # ==============================================================
            self._update_global_best()
            self.history.append(self.best_fitness)

            # ==============================================================
            # 9.  INERTIA DECAY  (multiplicative, per-iteration)
            # ==============================================================
            # w ← w * w_decay  shifts the balance from exploration (high w)
            # towards exploitation (low w) as the run progresses.
            w = w * cfg.w_decay

        return self.best_solution

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _get_best_index(self) -> int:
        """Return the index of the current best agent.

        Returns
        -------
        int
            Index into ``self.population`` / ``self.fitness``.
        """
        if self.conf.minimization:
            return int(np.argmin(self.fitness))
        return int(np.argmax(self.fitness))

    def _update_global_best(self) -> None:
        """Refresh ``best_solution`` and ``best_fitness`` when a strictly
        better agent exists in the current population.

        The position array is *copied* so that subsequent in-place
        population updates cannot silently corrupt the stored optimum.
        """
        idx               = self._get_best_index()
        candidate_fitness = float(self.fitness[idx])

        no_best_yet = not hasattr(self, "best_fitness")
        is_better   = (
            (    self.conf.minimization and candidate_fitness < self.best_fitness) or
            (not self.conf.minimization and candidate_fitness > self.best_fitness)
        ) if not no_best_yet else False

        if no_best_yet or is_better:
            self.best_fitness  = candidate_fitness
            self.best_solution = self.population[idx].copy()
