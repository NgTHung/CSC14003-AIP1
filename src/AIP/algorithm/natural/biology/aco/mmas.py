"""MAX-MIN Ant System (MMAS) for combinatorial optimization.

Extends Ant System with bounded pheromone trails [tau_min, tau_max] and
best-ant-only pheromone updates.  Prevents premature convergence by
limiting the pheromone range and optionally re-initializing trails.

Supports two solution types through ``DiscreteProblem.solution_type``:

* **Permutation** (e.g. TSP): edge-based pheromone ``tau[from][to]``.
* **Assignment** (e.g. Knapsack, Graph Coloring): position-value pheromone
  ``tau[position][value]``.

Reference: Stützle, T. & Hoos, H.H. (2000). MAX-MIN Ant System. Future
Generation Computer Systems, 16(8), 889-914.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from AIP.problems import DiscreteProblem
from AIP.algorithm import Algorithm


@dataclass
class MMASParameter:
    """Configuration parameters for MMAS.

    Attributes
    ----------
    rho : float
        Evaporation rate (0 < rho < 1). Typical: 0.02-0.2.
    m : int
        Number of ants in the colony.
    alpha : float
        Pheromone importance (α). Typical: 1.0.
    beta : float
        Heuristic importance (β). Typical: 2.0-5.0.
    cycle : int
        Number of iterations.
    p_best : float
        Probability that the best solution is constructed when
        the system has converged. Used to compute tau_min.
        Typical: 0.05.
    use_iteration_best : bool
        If True, alternate between iteration-best and global-best
        for pheromone deposits. If False, always use global-best.
    """

    rho: float
    m: int
    alpha: float
    beta: float
    cycle: int
    p_best: float = 0.05
    use_iteration_best: bool = True


class MMAS(Algorithm[DiscreteProblem, np.ndarray | None, float, MMASParameter]):
    """MAX-MIN Ant System for discrete optimization.

    Key differences from Ant System:
    - Pheromone trails are bounded within [tau_min, tau_max].
    - Only the best ant (iteration-best or global-best) deposits pheromone.
    - Trails are initialized to tau_max for maximum exploration.
    - Pheromone re-initialization when stagnation is detected.

    Automatically adapts to permutation or assignment problems based on
    ``problem.solution_type``.
    """

    tau: np.ndarray
    eta: np.ndarray
    tau_max: float
    tau_min: float
    _is_permutation: bool
    _combined: np.ndarray  # pre-computed tau^alpha * eta^beta

    def __init__(self, configuration: MMASParameter, problem: DiscreteProblem):
        """Initialize MMAS.

        Parameters
        ----------
        configuration : MMASParameter
            Algorithm hyperparameters.
        problem : DiscreteProblem
            Discrete optimization problem to solve.
        """
        super().__init__(configuration, problem)
        n = problem.n_dims
        self._is_permutation = problem.solution_type == "permutation"

        # Initial bounds (will be updated once best solution is found)
        self.tau_max = 1.0
        self.tau_min = self.tau_max / (2.0 * n)

        # Load problem-specific heuristic (e.g. 1/distance for TSP,
        # value/weight for Knapsack).  Falls back to all-ones when the
        # problem does not supply one.
        heuristic = problem.aco_heuristic()

        if self._is_permutation:
            self.tau = np.full((n, n), self.tau_max)
            self.eta = heuristic if heuristic is not None else np.ones((n, n))
        else:
            d = problem.domain_size
            self.tau = np.full((n, d), self.tau_max)
            self.eta = heuristic if heuristic is not None else np.ones((n, d))

        # Pre-compute eta^beta (constant across iterations)
        self._eta_beta = self.eta**self.conf.beta
        self._update_combined()

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

        self._stagnation_counter = 0
        self._last_improvement = 0

    def _update_combined(self):
        """Recompute the combined score matrix tau^alpha * eta^beta."""
        self._combined = self.tau**self.conf.alpha * self._eta_beta

    def _update_bounds(self):
        """Recompute tau_max and tau_min based on best-so-far cost.

        tau_max = 1 / (rho * L_best)
        tau_min = tau_max * (1 - p_best^(1/n)) / ((n/2 - 1) * p_best^(1/n))
        """
        if self.best_fitness == float("inf") or self.best_fitness <= 0:
            return

        self.tau_max = 1.0 / (self.conf.rho * self.best_fitness)

        n = self.problem.n_dims
        p_root = self.conf.p_best ** (1.0 / n)
        denominator = (n / 2.0 - 1.0) * p_root
        if denominator > 0:
            self.tau_min = self.tau_max * (1.0 - p_root) / denominator
        else:
            self.tau_min = self.tau_max / (2.0 * n)

        self.tau_min = min(self.tau_min, self.tau_max)

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution."""
        if self._is_permutation:
            return self._construct_permutation()
        return self._construct_assignment()

    def _construct_permutation(self) -> np.ndarray:
        """Build a permutation by visiting each node exactly once.

        Uses a boolean mask instead of a set for O(1) numpy indexing.
        """
        n = self.problem.n_dims
        visited = np.zeros(n, dtype=bool)
        solution = np.empty(n, dtype=np.intp)

        current = np.random.randint(0, n)
        solution[0] = current
        visited[current] = True

        for step in range(1, n):
            scores = self._combined[current].copy()
            scores[visited] = 0.0

            total = scores.sum()
            if total == 0.0:
                candidates = np.flatnonzero(~visited)
                current = candidates[np.random.randint(len(candidates))]
            else:
                scores /= total
                current = np.random.choice(n, p=scores)

            solution[step] = current
            visited[current] = True

        return solution.astype(float)

    def _construct_assignment(self) -> np.ndarray:
        """Build an assignment by choosing a value for each position.

        Vectorised: computes cumulative probabilities for all positions
        at once and samples via uniform random draws.
        """
        scores = self._combined.copy()
        totals = scores.sum(axis=1, keepdims=True)

        zero_mask = totals.ravel() == 0
        if zero_mask.any():
            scores[zero_mask] = 1.0
            totals[zero_mask] = scores.shape[1]

        cumprobs = np.cumsum(scores / totals, axis=1)
        rands = np.random.rand(scores.shape[0])
        solution = np.array(
            [
                np.searchsorted(cumprobs[pos], rands[pos])
                for pos in range(scores.shape[0])
            ],
            dtype=float,
        )

        return solution

    def construct_solution(self) -> list[tuple[np.ndarray, float]]:
        """Construct solutions for all ants.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            (solution, fitness) pairs for each ant.
        """
        self._update_combined()
        solutions: list[tuple[np.ndarray, float]] = []

        for _ in range(self.conf.m):
            solution = self._construct_ant_solution()
            fitness = cast(float, self.problem.eval(solution))
            solutions.append((solution, fitness))

        return solutions

    def pheromone_update(
        self,
        solutions: list[tuple[np.ndarray, float]],
        iteration: int,
    ):
        """Update pheromone using best-ant-only deposit and clamping.

        Parameters
        ----------
        solutions : list[tuple[np.ndarray, float]]
            (solution, fitness) pairs from all ants.
        iteration : int
            Current iteration number (for alternating update strategy).
        """
        # Find iteration-best
        iter_best_sol, iter_best_fit = min(solutions, key=lambda x: x[1])
        improved = False

        # Update global best
        if iter_best_fit < self.best_fitness:
            self.best_fitness = iter_best_fit
            self.best_solution = iter_best_sol.copy()
            self._update_bounds()
            self._last_improvement = iteration
            improved = True

        # Choose which ant deposits pheromone
        if self.conf.use_iteration_best and iteration % 2 == 0:
            deposit_sol = iter_best_sol
            deposit_fit = iter_best_fit
        else:
            deposit_sol = cast(np.ndarray, self.best_solution)
            deposit_fit = self.best_fitness

        # Evaporation
        self.tau *= 1 - self.conf.rho

        # Deposit
        deposit = 1.0 / deposit_fit if deposit_fit > 0 else 1.0

        if self._is_permutation:
            sol = deposit_sol.astype(np.intp)
            frm = sol
            to = np.roll(sol, -1)
            self.tau[frm, to] += deposit
            self.tau[to, frm] += deposit
        else:
            positions = np.arange(self.problem.n_dims)
            vals = deposit_sol.astype(np.intp)
            self.tau[positions, vals] += deposit

        # Clamp to [tau_min, tau_max]
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

        # Re-initialize pheromone on stagnation
        if not improved:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0

        if self._stagnation_counter > self.problem.n_dims:
            if self._is_permutation:
                n = self.problem.n_dims
                self.tau = np.full((n, n), self.tau_max)
            else:
                n = self.problem.n_dims
                d = self.problem.domain_size
                self.tau = np.full((n, d), self.tau_max)
            self._stagnation_counter = 0

    @override
    def run(self) -> np.ndarray:
        """Execute the MMAS algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        for iteration in range(self.conf.cycle):
            solutions = self.construct_solution()
            self.pheromone_update(solutions, iteration)

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
